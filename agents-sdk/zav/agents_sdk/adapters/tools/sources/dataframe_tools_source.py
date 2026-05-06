import base64
import gc
import io
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from fastexcel import CalamineCellError
from zav.logging import logger
from zav.pydantic_compat import BaseModel, model_validator

from zav.agents_sdk.adapters.async_wrapper import asyncify
from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
    ChatCompletion,
    ChatCompletionSender,
)
from zav.agents_sdk.adapters.retrievers.zav_retriever import ZAVRetriever
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable

matplotlib.use("Agg")


class FilterOperation(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOTIN = "notin"
    CONTAINS = "contains"


class FilterCondition(BaseModel):
    column: str
    op: FilterOperation
    value: Optional[Any] = None
    stored_value_reference: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_value_or_stored_value_reference(cls, values):
        if values.get("value") is None and values.get("stored_value_reference") is None:
            raise ValueError("Either value or stored_value_reference must be provided.")
        if (
            values.get("value") is not None
            and values.get("stored_value_reference") is not None
        ):
            raise ValueError(
                "Only one of value or stored_value_reference" " must be provided."
            )
        return values


class SupportedContentType(str, Enum):
    EXCEL = "application/vnd.openxmlformats-officedocument" ".spreadsheetml.sheet"
    CSV = "text/csv"
    XLSX = "application/vnd.ms-excel"


SUPPORTED_CONTENT_TYPES = ",".join([ct.value for ct in SupportedContentType])

SUPPORTED_FILTER_OPERATIONS = ",".join([op.value for op in FilterOperation])

_SYSTEM_PROMPT_SECTION = """\
## Tabular data handling

You have access to dataframe tools for working with CSV and Excel files.

- When the user provides an Excel or CSV document, prefer the dataframe \
tools over searching inside the document.
- First use `load_dataframe` with the document_id to load the file, \
then use the other dataframe tools to explore and analyze it.
- Supported content types: Excel (.xlsx), CSV (.csv).
- You can chain operations: load → schema → filter → aggregate → plot.
- For plotting, the tools return images directly in the response.\
"""


class DataFrameToolsSourceConfiguration(BaseModel):
    enabled: bool = False


def _map_dtype_to_json_type(dtype):
    if dtype in [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ]:
        return "integer"
    elif dtype in [pl.Float32, pl.Float64]:
        return "number"
    elif dtype == pl.Boolean:
        return "boolean"
    elif dtype in [pl.Utf8, pl.Categorical, pl.Object]:
        return "string"
    elif dtype in [pl.Date, pl.Datetime, pl.Time]:
        return "string"
    else:
        return "unknown"


_UNNAMED_PATTERN = re.compile(r"(?i)^(__)?unnamed(__|\s|:)?(\d+)?$")


def _sanitize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    renames = {}
    for i, col in enumerate(df.columns):
        if _UNNAMED_PATTERN.match(col.strip()):
            renames[col] = f"Column {i + 1}"
    if renames:
        return df.rename(renames)
    return df


async def _convert_string_columns_to_numeric(
    df: pl.DataFrame,
) -> pl.DataFrame:
    logger.debug(
        f"Starting string-to-numeric conversion for"
        f" {df.height} rows, {len(df.columns)} columns"
    )
    conversion_exprs = []
    for col in df.columns:
        if df.schema[col] == pl.Utf8:
            try:
                sample_size = min(1000, df.height)
                test_series = df[col].head(sample_size).drop_nulls()
                if test_series.len() == 0:
                    conversion_exprs.append(pl.col(col))
                    continue
                try:
                    clean_series = test_series.str.strip_chars()
                    int_converted = clean_series.cast(pl.Int64, strict=False)
                    if int_converted.null_count() == 0:
                        conversion_exprs.append(
                            pl.col(col)
                            .str.strip_chars()
                            .cast(pl.Int64, strict=False)
                            .alias(col)
                        )
                        continue
                except Exception:
                    pass
                try:
                    clean_series = test_series.str.strip_chars()
                    float_converted = clean_series.cast(pl.Float64, strict=False)
                    if float_converted.null_count() == 0:
                        conversion_exprs.append(
                            pl.col(col)
                            .str.strip_chars()
                            .cast(pl.Float64, strict=False)
                            .alias(col)
                        )
                        continue
                except Exception:
                    pass
                conversion_exprs.append(pl.col(col))
            except Exception:
                conversion_exprs.append(pl.col(col))
        else:
            conversion_exprs.append(pl.col(col))
    converted_df = df.select(conversion_exprs)
    del conversion_exprs
    gc.collect()
    return converted_df


def _sample_for_plotting(df: pl.DataFrame, max_points: int = 10000) -> pl.DataFrame:
    if df.height <= max_points:
        return df
    sample_size = min(max_points, df.height)
    step = df.height // sample_size
    indices = list(range(0, df.height, step))[:sample_size]
    return df.with_row_index().filter(pl.col("index").is_in(indices)).drop("index")


def _validate_plot_columns(
    df: pl.DataFrame,
    columns: List[str],
    required_numeric: bool = False,
) -> Dict[str, Any]:
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        return {
            "status": "error",
            "message": (f"Columns not found: {', '.join(missing_cols)}"),
        }
    if required_numeric:
        non_numeric = [c for c in columns if not df.schema[c].is_numeric()]
        if non_numeric:
            return {
                "status": "error",
                "message": ("Columns must be numeric:" f" {', '.join(non_numeric)}"),
            }
    return {"status": "ok"}


class DataFrameToolsSource(ToolsSource):

    source_name = "dataframe_tools"

    def __init__(
        self,
        retriever: ZAVRetriever,
        dataframe_tools_source_configuration: DataFrameToolsSourceConfiguration,
    ):
        self.__retriever = retriever
        self.__config = dataframe_tools_source_configuration
        self.enabled = dataframe_tools_source_configuration.enabled
        self.__dataframes: Dict[str, pl.DataFrame] = {}
        self.__stored_values: Dict[str, Any] = {}

    async def get_tools(self) -> List[Tool]:
        return [
            Tool.from_callable(
                name="load_dataframe",
                executable=self.load_dataframe,
            ),
            Tool.from_callable(
                name="get_dataframe_schema",
                executable=self.get_dataframe_schema,
            ),
            Tool.from_callable(
                name="preview_dataframe",
                executable=self.preview_dataframe,
            ),
            Tool.from_callable(
                name="read_dataframe",
                executable=self.read_dataframe,
            ),
            Tool.from_callable(
                name="count_dataframe_rows",
                executable=self.count_dataframe_rows,
            ),
            Tool.from_callable(
                name="get_column_values",
                executable=self.get_column_values,
            ),
            Tool.from_callable(
                name="get_column_statistics",
                executable=self.get_column_statistics,
            ),
            Tool.from_callable(
                name="get_missing_value_counts",
                executable=self.get_missing_value_counts,
            ),
            Tool.from_callable(
                name="filter_dataframe_rows",
                executable=self.filter_dataframe_rows,
            ),
            Tool.from_callable(
                name="sort_dataframe_rows",
                executable=self.sort_dataframe_rows,
            ),
            Tool.from_callable(
                name="aggregate_column",
                executable=self.aggregate_column,
            ),
            Tool.from_callable(
                name="merge_dataframes",
                executable=self.merge_dataframes,
            ),
            Tool.from_callable(
                name="concatenate_dataframes",
                executable=self.concatenate_dataframes,
            ),
            Tool.from_callable(
                name="create_basic_plot",
                executable=self.create_basic_plot,
            ),
            Tool.from_callable(
                name="create_multi_series_plot",
                executable=self.create_multi_series_plot,
            ),
            Tool.from_callable(
                name="create_distribution_plot",
                executable=self.create_distribution_plot,
            ),
            Tool.from_callable(
                name="create_time_series_plot",
                executable=self.create_time_series_plot,
            ),
            Tool.from_callable(
                name="create_correlation_heatmap",
                executable=self.create_correlation_heatmap,
            ),
        ]

    async def to_prompt(self) -> str:
        if not self.enabled:
            return ""
        return _SYSTEM_PROMPT_SECTION

    async def __create_plot_image(self, fig) -> Dict[str, Any]:
        try:
            buffer = io.BytesIO()
            await asyncify(fig.savefig)(
                buffer,
                format="png",
                bbox_inches="tight",
                dpi=100,
                facecolor="white",
            )
            buffer.seek(0)
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            plt.close(fig)
            buffer.close()
            gc.collect()
            return {
                "image_base64": image_base64,
                "image_format": "png",
                "image_size_bytes": len(image_bytes),
            }
        except Exception as e:
            plt.close(fig)
            raise e

    async def __get_document_content_bytes(self, document_id: str) -> Optional[bytes]:
        long_document_id = document_id.split("_")[0]
        tabular_bytes = None
        try:
            tabular_bytes = await self.__retriever.get_content_asset(long_document_id)
        except Exception as e:
            logger.debug(f"Error getting content asset for" f" {long_document_id}: {e}")
        if not tabular_bytes:
            try:
                tabular_text = await self.__retriever.get_full_text(long_document_id)
                if not tabular_text:
                    return None
                tabular_bytes = tabular_text.encode("utf-8")
            except Exception as e:
                logger.debug(f"Error getting full text for" f" {long_document_id}: {e}")
                return None
        return tabular_bytes

    def __build_schema_data(
        self, df: pl.DataFrame, sample_size: int = 5
    ) -> List[Dict[str, Any]]:
        effective_sample = min(sample_size, df.height)
        samples_df = df.head(effective_sample)
        schema_data = []
        for col in df.columns:
            dtype = df.schema[col]
            json_type = _map_dtype_to_json_type(dtype)
            sample_values = [
                str(v) if v is not None else "NaN" for v in samples_df[col].to_list()
            ]
            schema_data.append(
                {
                    "column": col,
                    "type": json_type,
                    "sample_values": sample_values,
                }
            )
        return schema_data

    @streamable(
        running_text="Loading '{{ output_dataframe_name }}' as dataframe...",
        completed_text="'Dataframe '{{ output_dataframe_name }}' loaded.",
        params_transform=hide,
        response_transform=hide,
    )
    async def load_dataframe(
        self,
        document_id: str,
        output_dataframe_name: str,
        document_content_type: SupportedContentType,
    ) -> Dict[str, Any]:
        f"""Load a CSV/Excel document and store it as named DataFrame(s).

        For Excel files, all sheets are loaded as separate DataFrames
        with names output_dataframe_name__sheet_name.

        Args:
            document_id: Reference to the document.
            output_dataframe_name: A short, human-readable name for the
                stored dataframe(s) (e.g. "sales_data", "quarterly_budget").
            document_content_type: Content type. Must be one of
                {SUPPORTED_CONTENT_TYPES}.

        Returns:
            Dict with schema, row_count, and dataframe name(s).
        """
        try:
            if document_content_type not in [ct.value for ct in SupportedContentType]:
                return {
                    "status": "error",
                    "message": (
                        f"Unsupported content type:"
                        f" {document_content_type}."
                        f" Only {SUPPORTED_CONTENT_TYPES}"
                        " are supported."
                    ),
                }
            tabular_bytes = await self.__get_document_content_bytes(document_id)
            if not tabular_bytes:
                return {
                    "status": "error",
                    "message": ("No content found for document_id:" f" {document_id}"),
                }
            if document_content_type in (
                SupportedContentType.EXCEL,
                SupportedContentType.XLSX,
            ):
                return await self.__load_excel(tabular_bytes, output_dataframe_name)
            elif document_content_type == SupportedContentType.CSV:
                return await self.__load_csv(tabular_bytes, output_dataframe_name)
            else:
                return {
                    "status": "error",
                    "message": (
                        f"Unsupported content type:"
                        f" {document_content_type}."
                        f" Only {SUPPORTED_CONTENT_TYPES}"
                        " are supported."
                    ),
                }
        except Exception as e:
            logger.debug(f"Error loading dataframe: {e}")
            return {"status": "error", "message": str(e)}

    async def __load_excel(
        self, tabular_bytes: bytes, output_dataframe_name: str
    ) -> Dict[str, Any]:
        logger.info(f"Loading Excel file with {len(tabular_bytes)} bytes")
        try:
            sheets = await asyncify(pl.read_excel)(
                io.BytesIO(tabular_bytes),
                sheet_id=0,
                engine="calamine",
                raise_if_empty=False,
                infer_schema_length=100000,
            )
        except CalamineCellError as exc:
            logger.warning(
                "Excel contains formula errors, retrying with"
                f" all columns as strings: {exc}"
            )
            sheets = await asyncify(pl.read_excel)(
                io.BytesIO(tabular_bytes),
                sheet_id=0,
                engine="calamine",
                raise_if_empty=False,
                infer_schema_length=100000,
                read_options={"dtypes": "string"},
            )

        dataframes_info = {}
        sheet_to_dataframe_name = {}
        df_name = output_dataframe_name
        schema_data: List[Dict[str, Any]] = []
        sample_size = 5
        df = pl.DataFrame()

        for sheet_name, sheet_df in sheets.items():
            if sheet_df.height == 0:
                continue
            sheet_df = _sanitize_column_names(sheet_df)
            sheet_df = await _convert_string_columns_to_numeric(sheet_df)
            df_name = f"{output_dataframe_name}__{sheet_name}"
            self.__dataframes[df_name] = sheet_df
            df = sheet_df
            schema_data = self.__build_schema_data(sheet_df)
            sample_size = min(5, sheet_df.height)
            dataframes_info[sheet_name] = {
                "dataframe_name": df_name,
                "schema": schema_data,
                "row_count": sheet_df.height,
                "sample_count": sample_size,
            }
            sheet_to_dataframe_name[sheet_name] = df_name

        if len(sheets) == 1:
            del sheets
            gc.collect()
            return {
                "status": "ok",
                "dataframe_name": df_name,
                "schema": schema_data,
                "row_count": df.height,
                "sample_count": sample_size,
            }
        else:
            del sheets
            gc.collect()
            return {
                "status": "ok",
                "info": (
                    "Input file contained multiple sheets."
                    " Each sheet was loaded into a separate"
                    " dataframe stored as"
                    " {output_dataframe_name}__sheet_name."
                ),
                "dataframes": dataframes_info,
                "sheet_to_dataframe_name": sheet_to_dataframe_name,
            }

    async def __load_csv(
        self, tabular_bytes: bytes, output_dataframe_name: str
    ) -> Dict[str, Any]:
        logger.info(f"Loading CSV file with {len(tabular_bytes)} bytes")
        try:
            lazy_df = pl.scan_csv(
                io.BytesIO(tabular_bytes),
                infer_schema_length=100000,
                separator=",",
                has_header=True,
            )
            header_check_df = lazy_df.head(1).collect()
            unnamed_cols = [
                col for col in header_check_df.columns if "Unnamed" in str(col)
            ]
            if unnamed_cols:
                lazy_df = pl.scan_csv(
                    io.BytesIO(tabular_bytes),
                    has_header=False,
                    infer_schema_length=100000,
                    separator=",",
                )
                first_row_df = lazy_df.head(1).collect()
                if first_row_df.height > 0:
                    first_row = first_row_df.row(0)
                    col_names = [
                        (
                            str(name).strip()
                            if name is not None and str(name).strip()
                            else f"Column {i}"
                        )
                        for i, name in enumerate(first_row)
                    ]
                    lazy_df = lazy_df.slice(1, None)
                    df = lazy_df.collect()
                    df.columns = col_names
                else:
                    df = lazy_df.collect()
                    df.columns = [f"Column {i}" for i in range(len(df.columns))]
                del first_row_df, header_check_df
            else:
                df = lazy_df.collect()
                del header_check_df
            del lazy_df
        except Exception as e:
            logger.warning(
                "Lazy loading failed, falling back to eager" f" loading: {e}"
            )
            df = await asyncify(pl.read_csv)(
                io.BytesIO(tabular_bytes),
                infer_schema_length=100000,
                separator=",",
                has_header=True,
            )
            unnamed_cols = [col for col in df.columns if "Unnamed" in str(col)]
            if unnamed_cols:
                df = await asyncify(pl.read_csv)(
                    io.BytesIO(tabular_bytes),
                    has_header=False,
                    infer_schema_length=100000,
                    separator=",",
                )
                if df.height > 0:
                    first_row = df.row(0)
                    col_names = [
                        (
                            str(name).strip()
                            if name is not None and str(name).strip()
                            else f"Column {i}"
                        )
                        for i, name in enumerate(first_row)
                    ]
                    df.columns = col_names
                    df = df.slice(1, df.height - 1)
                else:
                    df.columns = [f"Column {i}" for i in range(len(df.columns))]

        df = _sanitize_column_names(df)
        df = await _convert_string_columns_to_numeric(df)
        self.__dataframes[output_dataframe_name] = df
        gc.collect()
        schema_data = self.__build_schema_data(df)
        return {
            "status": "ok",
            "dataframe_name": output_dataframe_name,
            "schema": schema_data,
            "row_count": df.height,
            "sample_count": min(5, df.height),
        }

    @streamable(
        running_text="Getting schema for '{{ input_dataframe_name }}'...",
        completed_text="Schema loaded for '{{ input_dataframe_name }}' ({{ row_count }} rows).",  # noqa: E501
        params_transform=hide,
        response_transform=hide,
    )
    def get_dataframe_schema(self, input_dataframe_name: str) -> Dict[str, Any]:
        """Returns the schema of a named DataFrame.

        Args:
            input_dataframe_name: Name of the dataframe.

        Returns:
            Dict with schema, row_count, and sample_count.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        schema_data = self.__build_schema_data(df)
        return {
            "status": "ok",
            "dataframe_name": input_dataframe_name,
            "schema": schema_data,
            "row_count": df.height,
            "sample_count": min(5, df.height),
        }

    @streamable(
        running_text="Previewing first {{ n_rows }} rows of '{{ input_dataframe_name }}'...",  # noqa: E501
        completed_text=(
            "Loaded preview of first {{ n_rows }} rows"
            " of '{{ input_dataframe_name }}'."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    def preview_dataframe(
        self, input_dataframe_name: str, n_rows: int = 5
    ) -> Dict[str, Any]:
        """Returns a preview of the first n rows of a DataFrame.

        Args:
            input_dataframe_name: Name of the dataframe.
            n_rows: Number of rows to preview (1-20).

        Returns:
            Dict with preview rows.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        if not (1 <= n_rows <= 20):
            return {
                "status": "error",
                "message": "n_rows must be between 1 and 20.",
            }
        preview_df = df.head(n_rows)
        return {
            "status": "ok",
            "dataframe_name": input_dataframe_name,
            "preview": preview_df.to_dicts(),
            "preview_row_count": preview_df.height,
            "total_row_count": df.height,
        }

    @streamable(
        running_text="Reading rows {{ from_row }} to {{ to_row }} from '{{ input_dataframe_name }}'...",  # noqa: E501
        completed_text="Read rows {{ from_row }} to {{ to_row }} from '{{ input_dataframe_name }}'.",  # noqa: E501
        params_transform=hide,
        response_transform=hide,
    )
    def read_dataframe(
        self,
        input_dataframe_name: str,
        from_row: int = 1,
        to_row: int = 100,
    ) -> Dict[str, Any]:
        """Read rows from a DataFrame with pagination.

        Args:
            input_dataframe_name: Name of the dataframe.
            from_row: Start row (1-based, inclusive).
            to_row: End row (inclusive). Max 500 rows at once.

        Returns:
            Dict with rows and row counts.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        if not (0 <= to_row - from_row <= 499):
            return {
                "status": "error",
                "message": (
                    "Cannot return more than 500 rows at once."
                    " Paginate using from_row and to_row."
                ),
            }
        offset = from_row - 1
        length = to_row - from_row + 1
        result_df = df.slice(offset, length)
        return {
            "status": "ok",
            "dataframe_name": input_dataframe_name,
            f"rows (from row {from_row} to row {to_row})": (result_df.to_dicts()),
            "returned_row_count": result_df.height,
            "total_row_count": df.height,
        }

    @streamable(
        running_text="Counting rows in '{{ input_dataframe_name }}'...",
        completed_text="Counted {{ row_count }} rows in '{{ input_dataframe_name }}'.",
        params_transform=hide,
        response_transform=hide,
    )
    def count_dataframe_rows(self, input_dataframe_name: str) -> Dict[str, Any]:
        """Returns the number of rows in a DataFrame.

        Args:
            input_dataframe_name: Name of the dataframe.

        Returns:
            Dict with row_count.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        return {
            "status": "ok",
            "dataframe_name": input_dataframe_name,
            "row_count": df.height,
        }

    @streamable(
        running_text="Getting values from '{{ column }}' in '{{ input_dataframe_name }}'...",  # noqa: E501
        completed_text=(
            "Found {{ total_count }} values in column"
            " '{{ column }}' of '{{ input_dataframe_name }}'."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    def get_column_values(
        self,
        input_dataframe_name: str,
        column: str,
        output_variable_name: str,
        distinct: bool = True,
        return_size: int = 100,
    ) -> Dict[str, Any]:
        """Get values from a column, optionally distinct.

        Args:
            input_dataframe_name: Name of the dataframe.
            column: Column to get values from.
            output_variable_name: Name to store values under
                for use in stored_value_reference.
            distinct: Return only distinct values.
            return_size: Max values to return (default 100).

        Returns:
            Dict with values and stored_value_reference.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        if column not in df.columns:
            return {
                "status": "error",
                "message": (
                    f"Column '{column}' not found in"
                    f" DataFrame '{input_dataframe_name}'."
                ),
            }
        if distinct:
            unique_values = df[column].drop_nulls().unique().to_list()
        else:
            unique_values = df[column].drop_nulls().to_list()
        self.__stored_values[output_variable_name] = unique_values
        return {
            "status": "ok",
            "dataframe_name": input_dataframe_name,
            "column": column,
            "distinct": distinct,
            f"values (up to {return_size})": (unique_values[:return_size]),
            "total_count": len(unique_values),
            "stored_value_reference": output_variable_name,
        }

    @streamable(
        running_text="Computing statistics for '{{ input_dataframe_name }}'...",
        completed_text="Statistics computed for '{{ input_dataframe_name }}'.",
        params_transform=hide,
        response_transform=hide,
    )
    def get_column_statistics(
        self,
        input_dataframe_name: str,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get statistics for numeric columns.

        Args:
            input_dataframe_name: Name of the dataframe.
            columns: Columns to get stats for. All numeric if None.

        Returns:
            Dict with statistics per column.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        collected_warnings = []
        numeric_cols_to_process = []
        if columns:
            non_existent = []
            non_numeric = []
            for col_name in columns:
                if col_name not in df.columns:
                    non_existent.append(col_name)
                elif not df.schema[col_name].is_numeric():
                    non_numeric.append(col_name)
                else:
                    numeric_cols_to_process.append(col_name)
            if non_existent:
                collected_warnings.append(
                    "Columns not found:" f" {', '.join(non_existent)}"
                )
            if non_numeric:
                collected_warnings.append(
                    "Non-numeric columns skipped:" f" {', '.join(non_numeric)}"
                )
            if not numeric_cols_to_process:
                response: Dict[str, Any] = {
                    "status": "error",
                    "message": (
                        "No valid numeric columns were" " specified to process."
                    ),
                }
                if collected_warnings:
                    response["warnings"] = collected_warnings
                return response
        else:
            numeric_cols_to_process = [
                col for col, dtype in df.schema.items() if dtype.is_numeric()
            ]
            if not numeric_cols_to_process:
                return {
                    "status": "error",
                    "message": ("No numeric columns found in the" " DataFrame."),
                }
        stats_data_list = []
        for col_name in numeric_cols_to_process:
            col_series = df[col_name]
            current_stats: Dict[str, Any] = {"column": col_name}
            if col_series.null_count() == col_series.len():
                current_stats.update(
                    {
                        "count": 0,
                        "mean": None,
                        "std": None,
                        "min": None,
                        "q1": None,
                        "median": None,
                        "q3": None,
                        "max": None,
                    }
                )
            else:
                desc = col_series.describe()
                desc_dict = {row[0]: row[1] for row in desc.rows()}
                current_stats["count"] = (
                    int(desc_dict.get("count", 0))
                    if desc_dict.get("count") is not None
                    else 0
                )
                current_stats["mean"] = desc_dict.get("mean")
                current_stats["std"] = desc_dict.get("std")
                current_stats["min"] = desc_dict.get("min")
                current_stats["q1"] = desc_dict.get("25%")
                current_stats["median"] = desc_dict.get("50%")
                current_stats["q3"] = desc_dict.get("75%")
                current_stats["max"] = desc_dict.get("max")
            stats_data_list.append(current_stats)
        output_dict = {
            "status": "ok",
            "dataframe_name": input_dataframe_name,
            "statistics": stats_data_list,
        }
        if collected_warnings:
            output_dict["warnings"] = collected_warnings
        return output_dict

    @streamable(
        running_text="Checking for missing values in '{{ input_dataframe_name }}'...",
        completed_text="Missing value check complete for '{{ input_dataframe_name }}'.",
        params_transform=hide,
        response_transform=hide,
    )
    def get_missing_value_counts(
        self,
        input_dataframe_name: str,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get missing value counts per column.

        Args:
            input_dataframe_name: Name of the dataframe.
            columns: Columns to check. All if None.

        Returns:
            Dict with missing_data counts.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        target_cols = df.columns if columns is None else columns
        missing_data = []
        not_found_cols = []
        for col in target_cols:
            if col not in df.columns:
                not_found_cols.append(col)
            else:
                missing_count = df[col].null_count()
                missing_data.append(
                    {
                        "column": col,
                        "missing_count": int(missing_count),
                    }
                )
        output_dict: Dict[str, Any] = {
            "status": "ok",
            "dataframe_name": input_dataframe_name,
        }
        if not_found_cols:
            output_dict["not_found_columns"] = not_found_cols
        if missing_data:
            output_dict["missing_data"] = missing_data
        if not missing_data:
            output_dict["info"] = "No missing values found in the specified" " columns."
        return output_dict

    @streamable(
        running_text="Filtering '{{ input_dataframe_name }}'...",
        completed_text="{{ filtered_row_count }} of {{ total_row_count }} rows matched in '{{ input_dataframe_name }}'.",  # noqa: E501
        params_transform=hide,
        response_transform=hide,
    )
    async def filter_dataframe_rows(  # noqa: C901
        self,
        input_dataframe_name: str,
        filters: List[FilterCondition],
        output_dataframe_name: str,
    ) -> Dict[str, Any]:
        f"""Filter rows based on conditions and store the result.

        Args:
            input_dataframe_name: Name of the input dataframe.
            filters: List of filter conditions. Supported ops:
                {SUPPORTED_FILTER_OPERATIONS}. Values can use
                stored_value_reference from get_column_values.
            output_dataframe_name: Name for the filtered result.

        Returns:
            Dict with filtered_row_count and total_row_count.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        try:
            mask = pl.Series([True] * df.height)
            for f in filters:
                col = f.column
                op = f.op
                if f.value:
                    val = f.value
                elif f.stored_value_reference:
                    val = self.__stored_values[f.stored_value_reference]
                else:
                    return {
                        "status": "error",
                        "message": (
                            "Either value or"
                            " stored_value_reference must"
                            " be provided."
                        ),
                    }
                if val is None or (isinstance(val, list) and not val):
                    return {
                        "status": "error",
                        "message": ("Filter value is required for" f" column '{col}'."),
                    }
                if col not in df.columns:
                    return {
                        "status": "error",
                        "message": (
                            f"Filter column '{col}' not found"
                            f" in DataFrame"
                            f" '{input_dataframe_name}'."
                        ),
                    }
                col_series = df[col]
                dtype = df.schema[col]
                if dtype in [pl.Date, pl.Datetime]:
                    date_format = "%Y-%m-%d"
                    if isinstance(val, list):
                        val = [
                            (
                                datetime.strptime(str(v), date_format).date()
                                if v is not None
                                else None
                            )
                            for v in val
                        ]
                        val = [v for v in val if v is not None]
                    else:
                        val = (
                            datetime.strptime(str(val), date_format).date()
                            if val is not None
                            else None
                        )
                elif dtype.is_numeric() and not isinstance(val, list):
                    try:
                        val = float(val)
                    except Exception:
                        pass
                if op == FilterOperation.EQ:
                    mask &= col_series == val
                elif op == FilterOperation.NEQ:
                    mask &= col_series != val
                elif op == FilterOperation.GT:
                    mask &= col_series > val
                elif op == FilterOperation.GTE:
                    mask &= col_series >= val
                elif op == FilterOperation.LT:
                    mask &= col_series < val
                elif op == FilterOperation.LTE:
                    mask &= col_series <= val
                elif op == FilterOperation.IN:
                    if not isinstance(val, list):
                        val = [val]
                    mask &= col_series.is_in(val)
                elif op == FilterOperation.NOTIN:
                    if not isinstance(val, list):
                        val = [val]
                    mask &= ~col_series.is_in(val)
                elif op == FilterOperation.CONTAINS:
                    s_val = str(
                        val[0] if isinstance(val, list) and len(val) > 0 else val
                    )
                    regex_val = f"(?i){s_val}"
                    mask &= col_series.cast(pl.Utf8).str.contains(regex_val)
                else:
                    return {
                        "status": "error",
                        "message": ("Unsupported filter operation:" f" {op}"),
                    }
            filtered_df = await asyncify(df.filter)(mask)
            self.__dataframes[output_dataframe_name] = filtered_df
            del mask
            gc.collect()
            return {
                "status": "ok",
                "input_dataframe_name": input_dataframe_name,
                "output_dataframe_name": output_dataframe_name,
                "filtered_row_count": filtered_df.height,
                "total_row_count": df.height,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @streamable(
        running_text="Sorting '{{ input_dataframe_name }}' by {{ sort_by }} {{ sort_order }}...",  # noqa: E501
        completed_text="Sorted '{{ input_dataframe_name }}' by {{ sort_by }} {{ sort_order }}. Stored in '{{ output_dataframe_name }}'.",  # noqa: E501
        params_transform=hide,
        response_transform=hide,
    )
    async def sort_dataframe_rows(
        self,
        input_dataframe_name: str,
        sort_by: str,
        sort_order: str,
        output_dataframe_name: str,
    ) -> Dict[str, Any]:
        """Sort rows and store the result.

        Args:
            input_dataframe_name: Name of the input dataframe.
            sort_by: Column to sort by.
            sort_order: 'asc' or 'desc'.
            output_dataframe_name: Name for the sorted result.

        Returns:
            Dict with sorted_row_count.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        if sort_by not in df.columns:
            return {
                "status": "error",
                "message": (
                    f"Sort column '{sort_by}' not found in"
                    f" DataFrame '{input_dataframe_name}'."
                ),
            }
        ascending = sort_order == "asc"
        try:
            sorted_df = await asyncify(df.sort)(sort_by, descending=not ascending)
            self.__dataframes[output_dataframe_name] = sorted_df
            gc.collect()
            return {
                "status": "ok",
                "input_dataframe_name": input_dataframe_name,
                "output_dataframe_name": output_dataframe_name,
                "sorted_row_count": sorted_df.height,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @streamable(
        running_text="Computing {{ operation }} on '{{ column }}' in '{{ input_dataframe_name }}'...",  # noqa: E501
        completed_text="Computed {{ operation }} on '{{ column }}' in '{{ input_dataframe_name }}'. Stored in '{{ output_dataframe_name }}'.",  # noqa: E501
        params_transform=hide,
        response_transform=hide,
    )
    async def aggregate_column(
        self,
        input_dataframe_name: str,
        column: str,
        operation: str,
        output_dataframe_name: str,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Aggregate a numeric column, optionally grouped.

        Args:
            input_dataframe_name: Name of the input dataframe.
            column: Column to aggregate.
            operation: 'sum', 'avg', 'min', 'max', or 'count'.
            output_dataframe_name: Name for the result.
            group_by: Optional column(s) to group by.

        Returns:
            Dict with aggregation result.
        """
        df = self.__dataframes.get(input_dataframe_name)
        if df is None:
            return {
                "status": "error",
                "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
            }
        valid_ops = {"sum", "avg", "min", "max", "count", "mean"}
        agg_op = "mean" if operation == "avg" else operation
        if agg_op not in valid_ops:
            return {
                "status": "error",
                "message": (
                    f"Invalid operation: '{operation}'." f" Must be one of {valid_ops}"
                ),
            }
        if column not in df.columns:
            return {
                "status": "error",
                "message": (
                    f"Aggregation column '{column}' not found"
                    f" in DataFrame"
                    f" '{input_dataframe_name}'."
                ),
            }
        if agg_op != "count" and not df.schema[column].is_numeric():
            return {
                "status": "error",
                "message": (
                    f"Column '{column}' must be numeric for"
                    f" operation '{operation}'."
                ),
            }
        try:
            if group_by:
                if isinstance(group_by, str):
                    group_by_cols = [group_by]
                else:
                    group_by_cols = group_by
                missing_cols = [c for c in group_by_cols if c not in df.columns]
                if missing_cols:
                    return {
                        "status": "error",
                        "message": (
                            f"Grouping column(s)"
                            f" {missing_cols} not found in"
                            f" DataFrame"
                            f" '{input_dataframe_name}'."
                        ),
                    }
                grouped = await asyncify(df.group_by)(group_by_cols)
                if agg_op == "count":
                    result = await asyncify(grouped.agg)(
                        pl.count(column).alias("count")
                    )
                else:
                    result = await asyncify(grouped.agg)(
                        getattr(pl.col(column), agg_op)().alias(operation)
                    )
                self.__dataframes[output_dataframe_name] = result
                del grouped
                gc.collect()
                return {
                    "status": "ok",
                    "operation": operation,
                    "input_dataframe_name": input_dataframe_name,
                    "output_dataframe_name": output_dataframe_name,
                    "group_by": group_by,
                    "total_number_of_rows": result.height,
                }
            else:
                if agg_op == "count":
                    result_scalar = await asyncify(df[column].count)()
                else:
                    result_scalar = await asyncify(getattr(df[column], agg_op))()
                result_df = pl.DataFrame({operation: [result_scalar]})
                self.__dataframes[output_dataframe_name] = result_df
                gc.collect()
                return {
                    "status": "ok",
                    "operation": operation,
                    "input_dataframe_name": input_dataframe_name,
                    "output_dataframe_name": output_dataframe_name,
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @streamable(
        running_text=(
            "Merging dataframes '{{ left_dataframe_name }}'"
            " and '{{ right_dataframe_name }}'..."
        ),
        completed_text=(
            "Merged '{{ left_dataframe_name }}' and"
            " '{{ right_dataframe_name }}'"
            " into '{{ output_dataframe_name }}'."
        ),
        response_transform=hide,
    )
    async def merge_dataframes(
        self,
        left_dataframe_name: str,
        right_dataframe_name: str,
        output_dataframe_name: str,
        on: Any,
        how: Literal[
            "inner",
            "left",
            "right",
            "full",
            "semi",
            "anti",
            "cross",
            "outer",
        ] = "inner",
        suffixes: tuple = ("_left", "_right"),
    ) -> Dict[str, Any]:
        """Merge two DataFrames by specified columns.

        Args:
            left_dataframe_name: Name of the left DataFrame.
            right_dataframe_name: Name of the right DataFrame.
            output_dataframe_name: Name for the merged result.
            on: Column(s) to join on.
            how: Join type. Default 'inner'.
            suffixes: Suffixes for overlapping columns.

        Returns:
            Dict with row_count and columns.
        """
        try:
            left_df = self.__dataframes.get(left_dataframe_name)
            right_df = self.__dataframes.get(right_dataframe_name)
            if left_df is None:
                return {
                    "status": "error",
                    "message": (
                        "Left DataFrame" f" '{left_dataframe_name}' not found."
                    ),
                }
            if right_df is None:
                return {
                    "status": "error",
                    "message": (
                        "Right DataFrame" f" '{right_dataframe_name}'" " not found."
                    ),
                }
            if isinstance(on, str):
                on_cols = [on]
            else:
                on_cols = list(on)
            for col in on_cols:
                if col not in left_df.columns:
                    return {
                        "status": "error",
                        "message": (f"Column '{col}' not found in" " left DataFrame."),
                    }
                if col not in right_df.columns:
                    return {
                        "status": "error",
                        "message": (f"Column '{col}' not found in" " right DataFrame."),
                    }
            left_suffix, right_suffix = suffixes
            overlap_cols = set(left_df.columns) & set(right_df.columns) - set(on_cols)
            if overlap_cols:
                left_renames = {
                    col: f"{col}{left_suffix}"
                    for col in overlap_cols
                    if col in left_df.columns
                }
                right_renames = {
                    col: f"{col}{right_suffix}"
                    for col in overlap_cols
                    if col in right_df.columns
                }
                left_df = left_df.rename(left_renames)
                right_df = right_df.rename(right_renames)
            merged_df = await asyncify(left_df.join)(right_df, on=on, how=how)
            self.__dataframes[output_dataframe_name] = merged_df
            del left_df, right_df
            gc.collect()
            return {
                "status": "ok",
                "output_dataframe_name": output_dataframe_name,
                "row_count": merged_df.height,
                "columns": list(merged_df.columns),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @streamable(
        running_text="Concatenating dataframes '{{ dataframe_names }}'...",
        completed_text=(
            "Concatenated '{{ dataframe_names }}'"
            " into '{{ output_dataframe_name }}'."
        ),
        response_transform=hide,
    )
    async def concatenate_dataframes(
        self,
        dataframe_names: List[str],
        output_dataframe_name: str,
        axis: int = 0,
        ignore_index: bool = True,
    ) -> Dict[str, Any]:
        """Concatenate multiple DataFrames.

        Args:
            dataframe_names: Names of DataFrames to concatenate.
            output_dataframe_name: Name for the result.
            axis: 0 for vertical, 1 for horizontal.
            ignore_index: Whether to ignore index.

        Returns:
            Dict with row_count and columns.
        """
        try:
            if axis not in [0, 1]:
                return {
                    "status": "error",
                    "message": "Invalid axis. Must be 0 or 1.",
                }
            dfs = []
            for name in dataframe_names:
                df = self.__dataframes.get(name)
                if df is None:
                    return {
                        "status": "error",
                        "message": (f"DataFrame '{name}' not found."),
                    }
                dfs.append(df)
            concat_df = await asyncify(pl.concat)(
                dfs,
                how="vertical" if axis == 0 else "horizontal",
            )
            self.__dataframes[output_dataframe_name] = concat_df
            del dfs
            gc.collect()
            return {
                "status": "ok",
                "output_dataframe_name": output_dataframe_name,
                "row_count": concat_df.height,
                "columns": list(concat_df.columns),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @streamable(
        running_text="Creating {{ plot_type }} plot from '{{ input_dataframe_name }}'...",  # noqa: E501
        completed_text=(
            "Created {{ plot_type }} plot" " from '{{ input_dataframe_name }}'."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def create_basic_plot(  # noqa: C901
        self,
        input_dataframe_name: str,
        plot_type: Literal["line", "bar", "scatter", "histogram", "box"],
        x_column: str,
        y_column: Optional[str] = None,
        title: Optional[str] = None,
        figure_size: tuple = (10, 6),
        color: Optional[str] = None,
    ) -> Union[Dict[str, Any], ChatCompletion]:
        """Create a basic plot from DataFrame data.

        Args:
            input_dataframe_name: Name of the dataframe.
            plot_type: 'line', 'bar', 'scatter', 'histogram', or 'box'.
            x_column: Column for x-axis.
            y_column: Column for y-axis (not needed for histogram).
            title: Plot title.
            figure_size: (width, height). Default (10, 6).
            color: Plot color.

        Returns:
            ChatCompletion with base64 PNG image, or error dict.
        """
        try:
            df = self.__dataframes.get(input_dataframe_name)
            if df is None:
                return {
                    "status": "error",
                    "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
                }
            required_cols = [x_column]
            if plot_type != "histogram" and y_column:
                required_cols.append(y_column)
            validation = _validate_plot_columns(df, required_cols)
            if validation["status"] == "error":
                return validation
            if plot_type == "histogram" and y_column is None:
                pass
            elif y_column:
                pass
            else:
                return {
                    "status": "error",
                    "message": (
                        "y_column is required for plot type" f" '{plot_type}'."
                    ),
                }
            plot_df = _sample_for_plotting(df)
            if isinstance(figure_size, str):
                try:
                    size_str = figure_size.strip("()")
                    width, height = map(float, size_str.split(","))
                    figure_size = (width, height)
                except Exception:
                    figure_size = (10, 6)
            fig, ax = plt.subplots(figsize=figure_size)
            x_data = plot_df[x_column].to_numpy()
            if plot_type == "histogram":
                await asyncify(ax.hist)(x_data, bins=30, color=color, alpha=0.7)
                ax.set_xlabel(x_column)
                ax.set_ylabel("Frequency")
            else:
                y_data = plot_df.select(y_column).to_numpy().flatten()
                if plot_type == "line":
                    await asyncify(ax.plot)(x_data, y_data, color=color)
                elif plot_type == "bar":
                    if x_data.dtype.kind in ["U", "S", "O"]:
                        unique_categories = plot_df[x_column].unique().to_list()
                        y_values = []
                        for cat in unique_categories:
                            cat_data = plot_df.filter(pl.col(x_column) == cat)
                            y_val = cat_data.select(y_column).mean().item()
                            y_values.append(float(y_val))
                        x_positions = list(range(len(unique_categories)))
                        await asyncify(ax.bar)(x_positions, y_values, color=color)
                        ax.set_xticks(x_positions)
                        ax.set_xticklabels(
                            unique_categories,
                            rotation=45,
                            ha="right",
                        )
                    else:
                        if x_data.dtype.kind not in ["i", "u", "f"]:
                            x_numeric = x_data.astype(float)
                        else:
                            x_numeric = x_data
                        if y_data.dtype.kind not in ["i", "u", "f"]:
                            y_numeric = y_data.astype(float)
                        else:
                            y_numeric = y_data
                        await asyncify(ax.bar)(x_numeric, y_numeric, color=color)
                elif plot_type == "scatter":
                    await asyncify(ax.scatter)(x_data, y_data, color=color, alpha=0.6)
                elif plot_type == "box":
                    unique_vals = plot_df[x_column].unique().to_list()
                    box_data = [
                        plot_df.filter(pl.col(x_column) == val)
                        .select(y_column)
                        .to_numpy()
                        .flatten()
                        for val in unique_vals
                    ]
                    await asyncify(ax.boxplot)(box_data)
                    ax.set_xticklabels(unique_vals)
                ax.set_xlabel(x_column)
                if y_column:
                    ax.set_ylabel(y_column)
            if title:
                ax.set_title(title)
            plt.tight_layout()
            image_result = await self.__create_plot_image(fig)
            plot_description = f"Created {plot_type} plot"
            if plot_type == "histogram":
                plot_description += f" showing distribution of '{x_column}'"
            else:
                plot_description += f" with '{x_column}' vs '{y_column}'"
            if title:
                plot_description += f" (titled: {title})"
            return ChatCompletion(
                sender=ChatCompletionSender.BOT,
                content=plot_description,
                image_uri=("data:image/png;base64," f"{image_result['image_base64']}"),
            )
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @streamable(
        running_text="Creating multi-series {{ plot_type }} plot from '{{ input_dataframe_name }}'...",  # noqa: E501
        completed_text=(
            "Created {{ plot_type }} plot" " from '{{ input_dataframe_name }}'."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def create_multi_series_plot(  # noqa: C901
        self,
        input_dataframe_name: str,
        plot_type: Literal["line", "bar", "scatter"],
        x_column: str,
        y_columns: List[str],
        title: Optional[str] = None,
        figure_size: tuple = (10, 6),
        colors: Optional[List[str]] = None,
        legend: bool = True,
    ) -> Union[Dict[str, Any], ChatCompletion]:
        """Create a multi-series plot with multiple y-columns.

        Args:
            input_dataframe_name: Name of the dataframe.
            plot_type: 'line', 'bar', or 'scatter'.
            x_column: Column for x-axis.
            y_columns: Columns for y-axis series.
            title: Plot title.
            figure_size: (width, height). Default (10, 6).
            colors: Colors for each series.
            legend: Whether to show legend.

        Returns:
            ChatCompletion with base64 PNG image, or error dict.
        """
        try:
            df = self.__dataframes.get(input_dataframe_name)
            if df is None:
                return {
                    "status": "error",
                    "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
                }
            if not y_columns:
                return {
                    "status": "error",
                    "message": ("At least one y_column is required."),
                }
            all_columns = [x_column] + y_columns
            validation = _validate_plot_columns(df, all_columns)
            if validation["status"] == "error":
                return validation
            validation = _validate_plot_columns(df, y_columns, required_numeric=True)
            if validation["status"] == "error":
                return validation
            plot_df = _sample_for_plotting(df)
            if isinstance(figure_size, str):
                try:
                    size_str = figure_size.strip("()")
                    width, height = map(float, size_str.split(","))
                    figure_size = (width, height)
                except Exception:
                    figure_size = (10, 6)
            fig, ax = plt.subplots(figsize=figure_size)
            x_data = plot_df[x_column].to_numpy()
            if plot_type == "line":
                for i, y_col in enumerate(y_columns):
                    y_data = plot_df.select(y_col).to_numpy().flatten()
                    plot_color = colors[i] if colors and i < len(colors) else None
                    await asyncify(ax.plot)(
                        x_data,
                        y_data,
                        label=y_col,
                        color=plot_color,
                        marker="o",
                        markersize=3,
                    )
            elif plot_type == "bar":
                if x_data.dtype.kind in ["U", "S", "O"]:
                    unique_categories = plot_df[x_column].unique().to_list()
                    bar_width = 0.8 / len(y_columns)
                    x_positions = list(range(len(unique_categories)))
                    for i, y_col in enumerate(y_columns):
                        y_values = []
                        for cat in unique_categories:
                            cat_data = plot_df.filter(pl.col(x_column) == cat)
                            if cat_data.height > 0:
                                y_val = cat_data.select(y_col).mean().item()
                                y_values.append(float(y_val))
                            else:
                                y_values.append(0.0)
                        offset = (i - len(y_columns) / 2 + 0.5) * bar_width
                        plot_color = colors[i] if colors and i < len(colors) else None
                        bar_positions = [x + offset for x in x_positions]
                        await asyncify(ax.bar)(
                            bar_positions,
                            y_values,
                            width=bar_width,
                            label=y_col,
                            color=plot_color,
                        )
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(
                        unique_categories,
                        rotation=45,
                        ha="right",
                    )
                else:
                    unique_x_vals = sorted(plot_df[x_column].unique().to_list())
                    x_range = max(unique_x_vals) - min(unique_x_vals)
                    bar_width = x_range / len(unique_x_vals) / len(y_columns) * 0.8
                    for i, y_col in enumerate(y_columns):
                        y_values = []
                        x_vals = []
                        for x_val in unique_x_vals:
                            val_data = plot_df.filter(pl.col(x_column) == x_val)
                            if val_data.height > 0:
                                y_val = val_data.select(y_col).mean().item()
                                y_values.append(float(y_val))
                                offset = (i - len(y_columns) / 2 + 0.5) * bar_width
                                x_vals.append(x_val + offset)
                        plot_color = colors[i] if colors and i < len(colors) else None
                        await asyncify(ax.bar)(
                            x_vals,
                            y_values,
                            width=bar_width,
                            label=y_col,
                            color=plot_color,
                        )
            elif plot_type == "scatter":
                for i, y_col in enumerate(y_columns):
                    y_data = plot_df.select(y_col).to_numpy().flatten()
                    plot_color = colors[i] if colors and i < len(colors) else None
                    await asyncify(ax.scatter)(
                        x_data,
                        y_data,
                        label=y_col,
                        color=plot_color,
                        alpha=0.6,
                        s=30,
                    )
            ax.set_xlabel(x_column)
            ax.set_ylabel("Values")
            if title:
                ax.set_title(title)
            else:
                y_cols_str = ", ".join(y_columns)
                ax.set_title(
                    f"{plot_type.title()} Plot:" f" {y_cols_str} vs {x_column}"
                )
            if legend and len(y_columns) > 1:
                ax.legend()
            plt.tight_layout()
            image_result = await self.__create_plot_image(fig)
            y_cols_str = ", ".join(y_columns)
            plot_description = (
                f"Created multi-series {plot_type} plot"
                f" showing '{y_cols_str}' vs '{x_column}'"
            )
            if title:
                plot_description += f" (titled: {title})"
            return ChatCompletion(
                sender=ChatCompletionSender.BOT,
                content=plot_description,
                image_uri=("data:image/png;base64," f"{image_result['image_base64']}"),
            )
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @streamable(
        running_text="Creating {{ plot_type }} distribution for '{{ column }}' in '{{ input_dataframe_name }}'...",  # noqa: E501
        completed_text=(
            "Created {{ plot_type }} distribution plot"
            " for '{{ column }}' in '{{ input_dataframe_name }}'."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def create_distribution_plot(
        self,
        input_dataframe_name: str,
        column: str,
        plot_type: Literal["histogram", "density", "box"],
        bins: Optional[int] = 30,
        title: Optional[str] = None,
        figure_size: tuple = (8, 6),
    ) -> Union[Dict[str, Any], ChatCompletion]:
        """Create a distribution plot for a single column.

        Args:
            input_dataframe_name: Name of the dataframe.
            column: Column to analyze.
            plot_type: 'histogram', 'density', or 'box'.
            bins: Number of bins for histogram.
            title: Plot title.
            figure_size: (width, height). Default (8, 6).

        Returns:
            ChatCompletion with base64 PNG image, or error dict.
        """
        try:
            df = self.__dataframes.get(input_dataframe_name)
            if df is None:
                return {
                    "status": "error",
                    "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
                }
            validation = _validate_plot_columns(df, [column])
            if validation["status"] == "error":
                return validation
            if plot_type == "density":
                validation = _validate_plot_columns(df, [column], required_numeric=True)
                if validation["status"] == "error":
                    return validation
            plot_df = _sample_for_plotting(df)
            data = plot_df[column].drop_nulls().to_numpy()
            if len(data) == 0:
                return {
                    "status": "error",
                    "message": ("No valid data found in column" f" '{column}'."),
                }
            fig, ax = plt.subplots(figsize=figure_size)
            if plot_type == "histogram":
                bins_val = bins if bins is not None else 30
                await asyncify(ax.hist)(
                    data,
                    bins=bins_val,
                    alpha=0.7,
                    edgecolor="black",
                )
                ax.set_ylabel("Frequency")
            elif plot_type == "density":
                await asyncify(sns.histplot)(
                    data,
                    kde=True,
                    stat="density",
                    ax=ax,
                    alpha=0.7,
                )
                ax.set_ylabel("Density")
            elif plot_type == "box":
                await asyncify(ax.boxplot)(data, vert=True)
                ax.set_ylabel(column)
            ax.set_xlabel(column)
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"{plot_type.title()} of {column}")
            plt.tight_layout()
            image_result = await self.__create_plot_image(fig)
            plot_description = (
                f"Created {plot_type} distribution plot" f" for '{column}'"
            )
            if title:
                plot_description += f" (titled: {title})"
            return ChatCompletion(
                sender=ChatCompletionSender.BOT,
                content=plot_description,
                image_uri=("data:image/png;base64," f"{image_result['image_base64']}"),
            )
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @streamable(
        running_text="Creating time series plot from '{{ input_dataframe_name }}'...",
        completed_text="Created time series plot from '{{ input_dataframe_name }}'.",
        params_transform=hide,
        response_transform=hide,
    )
    async def create_time_series_plot(
        self,
        input_dataframe_name: str,
        date_column: str,
        value_columns: List[str],
        title: Optional[str] = None,
        figure_size: tuple = (12, 6),
        resample_frequency: Optional[str] = None,
    ) -> Union[Dict[str, Any], ChatCompletion]:
        """Create a time series line plot.

        Args:
            input_dataframe_name: Name of the dataframe.
            date_column: Column with date/datetime values.
            value_columns: Columns to plot as time series.
            title: Plot title.
            figure_size: (width, height). Default (12, 6).
            resample_frequency: Resample freq (e.g. "D", "W", "M").

        Returns:
            ChatCompletion with base64 PNG image, or error dict.
        """
        try:
            df = self.__dataframes.get(input_dataframe_name)
            if df is None:
                return {
                    "status": "error",
                    "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
                }
            all_columns = [date_column] + value_columns
            validation = _validate_plot_columns(df, all_columns)
            if validation["status"] == "error":
                return validation
            validation = _validate_plot_columns(
                df, value_columns, required_numeric=True
            )
            if validation["status"] == "error":
                return validation
            date_dtype = df.schema[date_column]
            if date_dtype not in [pl.Date, pl.Datetime, pl.Utf8]:
                return {
                    "status": "error",
                    "message": (
                        f"Column '{date_column}' must be"
                        " date, datetime, or string type."
                    ),
                }
            plot_df = _sample_for_plotting(df)
            if date_dtype == pl.Utf8:
                try:
                    plot_df = plot_df.with_columns(
                        [pl.col(date_column).str.to_datetime().alias(date_column)]
                    )
                except Exception:
                    return {
                        "status": "error",
                        "message": (f"Cannot parse '{date_column}'" " as datetime."),
                    }
            plot_df = plot_df.sort(date_column)
            plot_df = plot_df.drop_nulls([date_column] + value_columns)
            if plot_df.height == 0:
                return {
                    "status": "error",
                    "message": ("No valid data found after removing" " null values."),
                }
            fig, ax = plt.subplots(figsize=figure_size)
            dates = plot_df[date_column].to_numpy()
            for col in value_columns:
                values = plot_df[col].to_numpy()
                await asyncify(ax.plot)(
                    dates,
                    values,
                    label=col,
                    marker="o",
                    markersize=2,
                )
            ax.set_xlabel(date_column)
            ax.set_ylabel("Values")
            if title:
                ax.set_title(title)
            else:
                ax.set_title("Time Series:" f" {', '.join(value_columns)}")
            if len(value_columns) > 1:
                ax.legend()
            fig.autofmt_xdate()
            plt.tight_layout()
            image_result = await self.__create_plot_image(fig)
            columns_str = ", ".join(value_columns)
            plot_description = (
                f"Created time series plot of" f" '{columns_str}' over '{date_column}'"
            )
            if title:
                plot_description += f" (titled: {title})"
            return ChatCompletion(
                sender=ChatCompletionSender.BOT,
                content=plot_description,
                image_uri=("data:image/png;base64," f"{image_result['image_base64']}"),
            )
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @streamable(
        running_text="Creating correlation heatmap for '{{ input_dataframe_name }}'...",
        completed_text="Created correlation heatmap for '{{ input_dataframe_name }}'.",
        params_transform=hide,
        response_transform=hide,
    )
    async def create_correlation_heatmap(
        self,
        input_dataframe_name: str,
        columns: Optional[List[str]] = None,
        color_scheme: str = "coolwarm",
        figure_size: tuple = (10, 8),
        show_values: bool = True,
    ) -> Union[Dict[str, Any], ChatCompletion]:
        """Create a correlation heatmap for numeric columns.

        Args:
            input_dataframe_name: Name of the dataframe.
            columns: Columns to include. All numeric if None.
            color_scheme: Color map. Default 'coolwarm'.
            figure_size: (width, height). Default (10, 8).
            show_values: Show values in cells.

        Returns:
            ChatCompletion with base64 PNG image, or error dict.
        """
        try:
            df = self.__dataframes.get(input_dataframe_name)
            if df is None:
                return {
                    "status": "error",
                    "message": (f"DataFrame '{input_dataframe_name}'" " not found."),
                }
            if columns is None:
                numeric_cols = [
                    col for col, dtype in df.schema.items() if dtype.is_numeric()
                ]
                if not numeric_cols:
                    return {
                        "status": "error",
                        "message": ("No numeric columns found."),
                    }
            else:
                validation = _validate_plot_columns(df, columns, required_numeric=True)
                if validation["status"] == "error":
                    return validation
                numeric_cols = columns
            if len(numeric_cols) < 2:
                return {
                    "status": "error",
                    "message": (
                        "At least 2 numeric columns are" " required for correlation."
                    ),
                }
            plot_df = _sample_for_plotting(df)
            corr_df = plot_df.select(numeric_cols).drop_nulls()
            if corr_df.height == 0:
                return {
                    "status": "error",
                    "message": ("No valid data found after removing" " null values."),
                }
            correlation_matrix = await asyncify(corr_df.corr)()
            corr_values = correlation_matrix.to_numpy()
            fig, ax = plt.subplots(figsize=figure_size)
            await asyncify(sns.heatmap)(
                corr_values,
                annot=show_values,
                cmap=color_scheme,
                center=0,
                square=True,
                xticklabels=numeric_cols,
                yticklabels=numeric_cols,
                ax=ax,
                fmt=".2f" if show_values else "",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title("Correlation Heatmap")
            plt.tight_layout()
            image_result = await self.__create_plot_image(fig)
            cols_str = ", ".join(numeric_cols)
            plot_description = "Created correlation heatmap for columns:" f" {cols_str}"
            return ChatCompletion(
                sender=ChatCompletionSender.BOT,
                content=plot_description,
                image_uri=("data:image/png;base64," f"{image_result['image_base64']}"),
            )
        except Exception as e:
            return {"status": "error", "message": str(e)}


class DataFrameToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        zav_retriever: ZAVRetriever,
        dataframe_tools_source_configuration: DataFrameToolsSourceConfiguration = (
            DataFrameToolsSourceConfiguration()
        ),
    ) -> DataFrameToolsSource:
        return DataFrameToolsSource(
            retriever=zav_retriever,
            dataframe_tools_source_configuration=dataframe_tools_source_configuration,
        )
