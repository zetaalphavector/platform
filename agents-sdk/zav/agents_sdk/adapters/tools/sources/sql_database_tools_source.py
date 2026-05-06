import asyncio
import functools
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from sqlalchemy import MetaData, Table, and_, create_engine, func, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Select
from sqlalchemy.sql.selectable import Subquery
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field, model_validator

from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool


class SQLFilterOperation(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOTIN = "notin"
    LIKE = "like"


class SQLFilterCondition(BaseModel):
    column: str
    op: SQLFilterOperation
    value: Optional[Any] = None
    stored_value_reference: Optional[str] = None

    @model_validator(mode="after")
    def validate_value_or_reference(self):
        has_value = self.value is not None
        has_reference = self.stored_value_reference is not None
        if not has_value and not has_reference:
            raise ValueError(
                "Either 'value' or 'stored_value_reference' must be provided."
            )
        if has_value and has_reference:
            raise ValueError(
                "Cannot provide both 'value' and 'stored_value_reference'."
            )
        return self


class OperationType(str, Enum):
    SELECT = "select"
    FILTER = "filter"
    SORT = "sort"
    AGGREGATE = "aggregate"
    JOIN = "join"
    SELECT_COLUMNS = "select_columns"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class AggregateOperation(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    COUNT = "count"


class JoinType(str, Enum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"


class FilterParams(BaseModel):
    filters: List[Dict[str, Any]]


class SortParams(BaseModel):
    sort_by: str
    sort_order: SortOrder


class AggregateParams(BaseModel):
    column: str
    operation: AggregateOperation
    group_by: Optional[Union[str, List[str]]] = None


class SelectColumnsParams(BaseModel):
    columns: List[str]


class JoinParams(BaseModel):
    on: Union[str, List[str]]
    how: JoinType = JoinType.INNER


class QueryPlan(BaseModel):
    """Stores a query plan for deferred execution."""

    sources: List[str]
    operation: OperationType
    filter_params: Optional[FilterParams] = None
    sort_params: Optional[SortParams] = None
    aggregate_params: Optional[AggregateParams] = None
    select_columns_params: Optional[SelectColumnsParams] = None
    join_params: Optional[JoinParams] = None

    @model_validator(mode="after")
    def validate_params(self):
        op = self.operation
        if op == OperationType.FILTER and not self.filter_params:
            raise ValueError("filter_params required for FILTER operation")
        if op == OperationType.SORT and not self.sort_params:
            raise ValueError("sort_params required for SORT operation")
        if op == OperationType.AGGREGATE and not self.aggregate_params:
            raise ValueError("aggregate_params required for AGGREGATE operation")
        if op == OperationType.SELECT_COLUMNS and not self.select_columns_params:
            raise ValueError(
                "select_columns_params required for SELECT_COLUMNS operation"
            )
        if op == OperationType.JOIN and not self.join_params:
            raise ValueError("join_params required for JOIN operation")
        return self


class SQLDatabaseToolsSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable SQL database tools source.")
    connection_string: str = ""
    allowed_tables: Optional[List[str]] = None
    allowed_schemas: Optional[List[str]] = None
    max_rows_per_query: int = 10000
    query_timeout_seconds: int = 30


def _create_sql_engine(config: SQLDatabaseToolsSourceConfiguration) -> Engine:
    connect_args: Dict[str, Any] = {}
    if config.connection_string.startswith("postgresql"):
        connect_args["connect_timeout"] = config.query_timeout_seconds

    return create_engine(
        config.connection_string,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args=connect_args,
    )


class SQLDatabaseToolsSource(ToolsSource):
    source_name = "sql_database_tools"

    def __init__(
        self,
        config: SQLDatabaseToolsSourceConfiguration,
        engine: Optional[Engine] = None,
    ):
        self.enabled = config.enabled and bool(config.connection_string)
        self.__config = config
        self.__engine = engine
        self.__metadata = MetaData()
        self.__stored_values: Dict[str, Any] = {}
        self.__query_plans: Dict[str, QueryPlan] = {}
        self.__cached_tables: Dict[str, Table] = {}

    async def get_tools(self) -> List[Tool]:
        def _in_thread(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, functools.partial(fn, *args, **kwargs)
                )

            return wrapper

        return [
            Tool.from_callable(
                name="sql_list_tables",
                executable=_in_thread(self.list_tables),
            ),
            Tool.from_callable(
                name="sql_get_table_schema",
                executable=_in_thread(self.get_table_schema),
            ),
            Tool.from_callable(
                name="sql_preview_data",
                executable=_in_thread(self.preview_data),
            ),
            Tool.from_callable(
                name="sql_read_data",
                executable=_in_thread(self.read_data),
            ),
            Tool.from_callable(
                name="sql_count_rows",
                executable=_in_thread(self.count_rows),
            ),
            Tool.from_callable(
                name="sql_get_row_by_pk",
                executable=_in_thread(self.get_row_by_pk),
            ),
            Tool.from_callable(
                name="sql_select_columns",
                executable=self.select_columns,
            ),
            Tool.from_callable(
                name="sql_get_column_values",
                executable=_in_thread(self.get_column_values),
            ),
            Tool.from_callable(
                name="sql_filter_table_rows",
                executable=self.filter_table_rows,
            ),
            Tool.from_callable(
                name="sql_sort_table_rows",
                executable=self.sort_table_rows,
            ),
            Tool.from_callable(
                name="sql_aggregate_column",
                executable=self.aggregate_column,
            ),
            Tool.from_callable(
                name="sql_join_tables",
                executable=self.join_tables,
            ),
            Tool.from_callable(
                name="sql_insert_row",
                executable=_in_thread(self.insert_row),
            ),
        ]

    def __validate_result_name(self, name: str) -> None:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Invalid result name: {name}. "
                "Must start with a letter or underscore and contain only "
                "alphanumeric characters and underscores."
            )

    def __parse_table_name(self, table_name: str) -> Tuple[Optional[str], str]:
        parts = table_name.split(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, table_name

    def __is_table_allowed(self, schema: Optional[str], table: str) -> bool:
        if self.__config.allowed_schemas is not None:
            if schema and schema not in self.__config.allowed_schemas:
                return False

        if self.__config.allowed_tables is not None:
            full_name = f"{schema}.{table}" if schema else table
            if full_name not in self.__config.allowed_tables and (
                table not in self.__config.allowed_tables
            ):
                return False

        return True

    def __validate_table_access(self, schema: Optional[str], table: str) -> None:
        if not self.__is_table_allowed(schema, table):
            full_name = f"{schema}.{table}" if schema else table
            raise ValueError(f"Access to table '{full_name}' is not allowed.")

    def __get_table(self, table_name: str) -> Table:
        if table_name in self.__cached_tables:
            return self.__cached_tables[table_name]

        schema, table = self.__parse_table_name(table_name)
        self.__validate_table_access(schema, table)

        table_obj = Table(
            table,
            self.__metadata,
            schema=schema,
            autoload_with=self.__engine,
        )
        self.__cached_tables[table_name] = table_obj
        return table_obj

    def __resolve_stored_value(self, filter_condition: SQLFilterCondition) -> Any:
        if filter_condition.stored_value_reference not in self.__stored_values:
            raise ValueError(
                f"Stored value '{filter_condition.stored_value_reference}' "
                "not found."
            )
        return self.__stored_values[filter_condition.stored_value_reference]

    def __build_filter_clause(
        self,
        source: Union[Table, Select, Subquery],
        filters: List[SQLFilterCondition],
    ) -> Any:
        conditions = []
        for f in filters:
            col = source.c[f.column]  # type: ignore[union-attr]
            value = (
                self.__resolve_stored_value(f) if f.stored_value_reference else f.value
            )

            if f.op == SQLFilterOperation.EQ:
                conditions.append(col == value)
            elif f.op == SQLFilterOperation.NEQ:
                conditions.append(col != value)
            elif f.op == SQLFilterOperation.GT:
                conditions.append(col > value)
            elif f.op == SQLFilterOperation.GTE:
                conditions.append(col >= value)
            elif f.op == SQLFilterOperation.LT:
                conditions.append(col < value)
            elif f.op == SQLFilterOperation.LTE:
                conditions.append(col <= value)
            elif f.op == SQLFilterOperation.IN:
                if value is None or not isinstance(value, (list, tuple)):
                    raise ValueError("IN operation requires a list or tuple value")
                conditions.append(col.in_(value))
            elif f.op == SQLFilterOperation.NOTIN:
                if value is None or not isinstance(value, (list, tuple)):
                    raise ValueError("NOTIN operation requires a list or tuple value")
                conditions.append(~col.in_(value))
            elif f.op == SQLFilterOperation.LIKE:
                conditions.append(col.like(value))
            else:
                raise ValueError(f"Unsupported filter operation: {f.op}")

        if not conditions:
            raise ValueError("At least one filter condition is required.")
        return and_(*conditions) if len(conditions) > 1 else conditions[0]

    def __build_aggregation_expr(
        self, col: Any, operation: AggregateOperation, column_name: str
    ) -> Any:
        if operation == AggregateOperation.SUM:
            return func.sum(col).label(f"sum_{column_name}")
        elif operation == AggregateOperation.MEAN:
            return func.avg(col).label(f"mean_{column_name}")
        elif operation == AggregateOperation.MIN:
            return func.min(col).label(f"min_{column_name}")
        elif operation == AggregateOperation.MAX:
            return func.max(col).label(f"max_{column_name}")
        elif operation == AggregateOperation.COUNT:
            return func.count(col).label(f"count_{column_name}")
        else:
            raise ValueError(f"Unsupported aggregation operation: {operation}")

    def __build_join(
        self,
        source_table: Union[Table, Select],
        right_table: Union[Table, Select],
        on_cols: List[str],
        how: JoinType,
    ) -> Select:
        left_source: Union[Table, Subquery] = (
            source_table if isinstance(source_table, Table) else source_table.subquery()
        )
        right_source: Union[Table, Subquery] = (
            right_table if isinstance(right_table, Table) else right_table.subquery()
        )

        join_condition = and_(
            *[left_source.c[col] == right_source.c[col] for col in on_cols]
        )

        if how == JoinType.INNER:
            joined = left_source.join(right_source, join_condition)
        elif how == JoinType.LEFT:
            joined = left_source.join(right_source, join_condition, isouter=True)
        elif how == JoinType.RIGHT:
            joined = right_source.join(left_source, join_condition, isouter=True)
        elif how == JoinType.FULL:
            joined = left_source.join(right_source, join_condition, full=True)
        else:
            raise ValueError(f"Unsupported join type: {how}")

        return select(left_source, right_source).select_from(joined)

    def __plan_to_cte(self, plan: QueryPlan, cte_refs: Dict[str, Select]) -> Select:
        source_name = plan.sources[0]
        if source_name not in self.__query_plans:
            source_table: Union[Table, Select] = self.__get_table(source_name)
        else:
            source_table = cte_refs[source_name]

        if plan.operation == OperationType.SELECT:
            if isinstance(source_table, Table):
                return select(source_table)
            return source_table

        elif plan.operation == OperationType.FILTER:
            filter_params = cast(FilterParams, plan.filter_params)
            filters = [SQLFilterCondition(**f) for f in filter_params.filters]
            if isinstance(source_table, Table):
                where_clause = self.__build_filter_clause(source_table, filters)
                return select(source_table).where(where_clause)
            else:
                subq = source_table.subquery()
                where_clause = self.__build_filter_clause(subq, filters)
                return select(subq).where(where_clause)

        elif plan.operation == OperationType.SORT:
            sort_params = cast(SortParams, plan.sort_params)
            if isinstance(source_table, Table):
                sort_col = source_table.c[sort_params.sort_by]
                order_expr = (
                    sort_col.desc()
                    if sort_params.sort_order == SortOrder.DESC
                    else sort_col.asc()
                )
                return select(source_table).order_by(order_expr)
            else:
                subq = source_table.subquery()
                subq_sort_col = subq.c[sort_params.sort_by]
                order_expr = (
                    subq_sort_col.desc()
                    if sort_params.sort_order == SortOrder.DESC
                    else subq_sort_col.asc()
                )
                return select(subq).order_by(order_expr)

        elif plan.operation == OperationType.SELECT_COLUMNS:
            select_columns_params = cast(
                SelectColumnsParams, plan.select_columns_params
            )
            if isinstance(source_table, Table):
                return select(
                    *[source_table.c[c] for c in select_columns_params.columns]
                )
            else:
                subq = source_table.subquery()
                return select(*[subq.c[c] for c in select_columns_params.columns])

        elif plan.operation == OperationType.AGGREGATE:
            aggregate_params = cast(AggregateParams, plan.aggregate_params)
            source: Union[Table, Subquery] = (
                source_table
                if isinstance(source_table, Table)
                else source_table.subquery()
            )

            col = source.c[aggregate_params.column]
            agg_expr = self.__build_aggregation_expr(
                col,
                aggregate_params.operation,
                aggregate_params.column,
            )

            group_by = aggregate_params.group_by
            if group_by:
                if isinstance(group_by, str):
                    group_by = [group_by]
                group_cols = [source.c[g] for g in group_by]
                return select(*group_cols, agg_expr).group_by(*group_cols)
            return select(agg_expr)

        elif plan.operation == OperationType.JOIN:
            join_params = cast(JoinParams, plan.join_params)
            right_name = plan.sources[1]
            if right_name not in self.__query_plans:
                right_table: Union[Table, Select] = self.__get_table(right_name)
            else:
                right_table = cte_refs[right_name]

            on_cols = join_params.on
            if isinstance(on_cols, str):
                on_cols = [on_cols]

            return self.__build_join(
                source_table, right_table, on_cols, join_params.how
            )

        raise ValueError(f"Unknown operation type: {plan.operation}")

    def __execute_query_plan(
        self, engine: Engine, target: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if target not in self.__query_plans:
            table = self.__get_table(target)
            query = select(table)
            if limit:
                query = query.limit(limit)
            with engine.connect() as conn:
                result = conn.execute(query)
                return [dict(row._mapping) for row in result]

        plans = list(self.__query_plans.keys())
        cte_refs: Dict[str, Select] = {}

        for name in plans:
            plan = self.__query_plans[name]
            query = self.__plan_to_cte(plan, cte_refs)

            if name == target:
                if limit:
                    query = query.limit(limit)
                with engine.connect() as conn:
                    result = conn.execute(query)
                    return [dict(row._mapping) for row in result]

            cte_refs[name] = query

        return []

    def list_tables(self, schema: Optional[str] = None) -> Dict[str, Any]:
        """List tables with schema info.

        Args:
            schema: Optional schema to filter by.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - tables (list): List of {schema, table} dicts.
                - message (str): Error message if status is 'error'.
        """
        if self.__engine is None:
            return {
                "status": "error",
                "message": "SQL database engine is not configured.",
            }
        try:
            inspector = inspect(self.__engine)
            tables = []

            schemas_to_check = [schema] if schema else inspector.get_schema_names()

            for s in schemas_to_check:
                for table_name in inspector.get_table_names(schema=s):
                    if self.__is_table_allowed(s, table_name):
                        tables.append({"schema": s, "table": table_name})

            return {"status": "ok", "tables": tables}

        except Exception as e:
            logger.debug(f"Error listing tables: {e}")
            return {"status": "error", "message": str(e)}

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get column names, types, and sample values for a table.

        Args:
            table_name: Table name in 'schema.table' or 'table' format.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - table_name (str): The table name.
                - columns (list): List of column info dicts with name,
                  type, primary_key, and sample_values.
                - message (str): Error message if status is 'error'.
        """
        if self.__engine is None:
            return {
                "status": "error",
                "message": "SQL database engine is not configured.",
            }
        try:
            schema, table = self.__parse_table_name(table_name)
            self.__validate_table_access(schema, table)

            inspector = inspect(self.__engine)
            columns_info = inspector.get_columns(table, schema=schema)

            pk_info = inspector.get_pk_constraint(table, schema=schema)
            pk_columns = set(pk_info.get("constrained_columns", []))

            table_obj = self.__get_table(table_name)
            with self.__engine.connect() as conn:
                sample_query = select(table_obj).limit(5)
                sample_result = conn.execute(sample_query)
                sample_rows = [dict(row._mapping) for row in sample_result]

            columns = []
            for col_info in columns_info:
                col_name = col_info["name"]
                col_type = str(col_info["type"])
                sample_values = [
                    row.get(col_name) for row in sample_rows if col_name in row
                ][:3]

                columns.append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "is_primary_key": col_name in pk_columns,
                        "sample_values": sample_values,
                    }
                )

            return {
                "status": "ok",
                "table_name": table_name,
                "columns": columns,
            }

        except Exception as e:
            logger.debug(f"Error getting table schema: {e}")
            return {"status": "error", "message": str(e)}

    def preview_data(self, input_name: str, n_rows: int = 5) -> Dict[str, Any]:
        """Preview first N rows from a table or stored result.

        Args:
            input_name: Table name or stored result name.
            n_rows: Number of rows to preview (default 5, max 20).

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - input_name (str): The input name.
                - preview (list): List of row dicts.
                - preview_row_count (int): Number of rows in preview.
                - message (str): Error message if status is 'error'.
        """
        if self.__engine is None:
            return {
                "status": "error",
                "message": "SQL database engine is not configured.",
            }
        try:
            if n_rows < 1 or n_rows > 20:
                return {
                    "status": "error",
                    "message": "n_rows must be between 1 and 20",
                }

            rows = self.__execute_query_plan(self.__engine, input_name, limit=n_rows)
            return {
                "status": "ok",
                "input_name": input_name,
                "preview": rows,
                "preview_row_count": len(rows),
            }

        except Exception as e:
            logger.debug(f"Error previewing data: {e}")
            return {"status": "error", "message": str(e)}

    def read_data(
        self, input_name: str, from_row: int = 0, to_row: int = 100
    ) -> Dict[str, Any]:
        """Read rows with pagination from a table or stored result.

        Args:
            input_name: Table name or stored result name.
            from_row: Starting row (0-indexed).
            to_row: Ending row (exclusive).

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - input_name (str): The input name.
                - rows (list): List of row dicts.
                - returned_row_count (int): Number of rows returned.
                - message (str): Error message if status is 'error'.
        """
        if self.__engine is None:
            return {
                "status": "error",
                "message": "SQL database engine is not configured.",
            }
        try:
            max_rows = min(
                to_row - from_row,
                self.__config.max_rows_per_query,
                500,
            )
            if max_rows <= 0:
                return {
                    "status": "error",
                    "message": "Invalid row range.",
                }

            if input_name not in self.__query_plans:
                table = self.__get_table(input_name)
                query = select(table).offset(from_row).limit(max_rows)

                with self.__engine.connect() as conn:
                    result = conn.execute(query)
                    rows = [dict(row._mapping) for row in result]
            else:
                all_rows = self.__execute_query_plan(
                    self.__engine, input_name, limit=to_row
                )
                rows = all_rows[from_row:to_row]

            return {
                "status": "ok",
                "input_name": input_name,
                "rows": rows,
                "returned_row_count": len(rows),
                "from_row": from_row,
                "to_row": from_row + len(rows),
            }

        except Exception as e:
            logger.debug(f"Error reading data: {e}")
            return {"status": "error", "message": str(e)}

    def count_rows(
        self,
        input_name: str,
        filters: Optional[List[SQLFilterCondition]] = None,
    ) -> Dict[str, Any]:
        """Count rows in a table or stored result, optionally with
        filters.

        Args:
            input_name: Table name or stored result name.
            filters: Optional list of filter conditions.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - input_name (str): The input name.
                - row_count (int): Number of rows.
                - message (str): Error message if status is 'error'.
        """
        if self.__engine is None:
            return {
                "status": "error",
                "message": "SQL database engine is not configured.",
            }
        try:
            if input_name not in self.__query_plans:
                table = self.__get_table(input_name)
                query = select(func.count()).select_from(table)

                if filters:
                    where_clause = self.__build_filter_clause(table, filters)
                    query = query.where(where_clause)

                with self.__engine.connect() as conn:
                    row_count = conn.execute(query).scalar()
            else:
                rows = self.__execute_query_plan(self.__engine, input_name)
                row_count = len(rows)

            return {"status": "ok", "row_count": row_count}

        except Exception as e:
            logger.debug(f"Error counting rows: {e}")
            return {"status": "error", "message": str(e)}

    def get_row_by_pk(self, table_name: str, pk: Dict[str, Any]) -> Dict[str, Any]:
        """Get a single row by its primary key.

        Args:
            table_name: Table name in 'schema.table' or 'table' format.
            pk: Dictionary mapping primary key column names to values.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - table_name (str): The table name.
                - row (dict): The row data, or None if not found.
                - message (str): Error message if status is 'error'.
        """
        if self.__engine is None:
            return {
                "status": "error",
                "message": "SQL database engine is not configured.",
            }
        try:
            table = self.__get_table(table_name)

            filters = [
                SQLFilterCondition(column=col, op=SQLFilterOperation.EQ, value=val)
                for col, val in pk.items()
            ]

            where_clause = self.__build_filter_clause(table, filters)
            query = select(table).where(where_clause).limit(1)

            with self.__engine.connect() as conn:
                result = conn.execute(query)
                row = result.fetchone()
                if row:
                    return {
                        "status": "ok",
                        "row": dict(row._mapping),
                    }
                else:
                    return {"status": "ok", "row": None}

        except Exception as e:
            logger.debug(f"Error getting row by PK: {e}")
            return {"status": "error", "message": str(e)}

    def select_columns(
        self,
        input_name: str,
        columns: List[str],
        output_result_name: str,
    ) -> Dict[str, Any]:
        """Select specific columns from a table or stored result.

        Args:
            input_name: Table name or stored result name.
            columns: List of column names to select.
            output_result_name: Name to store the result under.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - input_name (str): The input name.
                - output_result_name (str): The output result name.
                - columns (list): Selected columns.
                - message (str): Error message if status is 'error'.
        """
        try:
            self.__validate_result_name(output_result_name)

            self.__query_plans[output_result_name] = QueryPlan(
                sources=[input_name],
                operation=OperationType.SELECT_COLUMNS,
                select_columns_params=SelectColumnsParams(columns=columns),
            )

            return {
                "status": "ok",
                "input_name": input_name,
                "output_result_name": output_result_name,
                "columns": columns,
            }

        except Exception as e:
            logger.debug(f"Error selecting columns: {e}")
            return {"status": "error", "message": str(e)}

    def get_column_values(
        self,
        input_name: str,
        column: str,
        output_variable_name: str,
        distinct: bool = False,
        filters: Optional[List[SQLFilterCondition]] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get distinct or all values from a column.

        Args:
            input_name: Table name or stored result name.
            column: Column name to get values from.
            output_variable_name: Name to store the values under.
            distinct: Whether to return only distinct values.
            limit: Maximum number of values to return.
            filters: Optional list of filter conditions.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - input_name (str): The input name.
                - column (str): The column name.
                - values (list): List of values.
                - total_count (int): Total count of values.
                - stored_value_reference (str): Name of stored values.
                - message (str): Error message if status is 'error'.
        """
        if self.__engine is None:
            return {
                "status": "error",
                "message": "SQL database engine is not configured.",
            }
        try:
            self.__validate_result_name(output_variable_name)

            if input_name not in self.__query_plans:
                table = self.__get_table(input_name)
                col = table.c[column]

                if distinct:
                    query = select(col.distinct())
                else:
                    query = select(col)

                if filters:
                    where_clause = self.__build_filter_clause(table, filters)
                    query = query.where(where_clause)

                query = query.limit(limit)

                with self.__engine.connect() as conn:
                    result = conn.execute(query)
                    values = [row[0] for row in result]

                count_query = select(func.count(col.distinct() if distinct else col))
                if filters:
                    where_clause = self.__build_filter_clause(table, filters)
                    count_query = count_query.where(where_clause)

                with self.__engine.connect() as conn:
                    total_count = conn.execute(count_query).scalar()
            else:
                rows = self.__execute_query_plan(self.__engine, input_name)
                all_values = [row[column] for row in rows if column in row]

                if distinct:
                    values = list(set(all_values))[:limit]
                else:
                    values = all_values[:limit]
                total_count = len(all_values)

            self.__stored_values[output_variable_name] = values

            return {
                "status": "ok",
                "stored_value_reference": output_variable_name,
                "values (up to 100)": values,
                "total_count": total_count,
            }

        except Exception as e:
            logger.debug(f"Error getting column values: {e}")
            return {"status": "error", "message": str(e)}

    def filter_table_rows(
        self,
        input_name: str,
        filters: List[SQLFilterCondition],
        output_result_name: str,
    ) -> Dict[str, Any]:
        """Filter rows based on structured conditions.

        Args:
            input_name: Table name or stored result name.
            filters: List of filter conditions. Supported operations:
                eq, neq, gt, gte, lt, lte, in, notin.
            output_result_name: Name to store the filtered result under.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - input_name (str): The input name.
                - output_result_name (str): The output result name.
                - filters_applied (int): Number of filters applied.
                - message (str): Error message if status is 'error'.
        """
        try:
            self.__validate_result_name(output_result_name)

            self.__query_plans[output_result_name] = QueryPlan(
                sources=[input_name],
                operation=OperationType.FILTER,
                filter_params=FilterParams(filters=[f.model_dump() for f in filters]),
            )

            return {
                "status": "ok",
                "input_name": input_name,
                "output_result_name": output_result_name,
                "filters_applied": len(filters),
            }

        except Exception as e:
            logger.debug(f"Error filtering rows: {e}")
            return {"status": "error", "message": str(e)}

    def sort_table_rows(
        self,
        input_name: str,
        sort_by: str,
        sort_order: str,
        output_result_name: str,
    ) -> Dict[str, Any]:
        """Sort rows by column(s).

        Args:
            input_name: Table name or stored result name.
            sort_by: Column name to sort by.
            sort_order: Sort order ('asc' or 'desc').
            output_result_name: Name to store the sorted result under.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - input_name (str): The input name.
                - output_result_name (str): The output result name.
                - sort_by (str): Column sorted by.
                - sort_order (str): Sort order used.
                - message (str): Error message if status is 'error'.
        """
        try:
            self.__validate_result_name(output_result_name)

            try:
                sort_order_enum = SortOrder(sort_order)
            except ValueError:
                return {
                    "status": "error",
                    "message": "sort_order must be one of: "
                    f"{[e.value for e in SortOrder]}.",
                }

            self.__query_plans[output_result_name] = QueryPlan(
                sources=[input_name],
                operation=OperationType.SORT,
                sort_params=SortParams(sort_by=sort_by, sort_order=sort_order_enum),
            )

            return {
                "status": "ok",
                "input_name": input_name,
                "output_result_name": output_result_name,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }

        except Exception as e:
            logger.debug(f"Error sorting rows: {e}")
            return {"status": "error", "message": str(e)}

    def aggregate_column(
        self,
        input_name: str,
        column: str,
        operation: str,
        output_result_name: str,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Aggregate a column with optional GROUP BY.

        Args:
            input_name: Table name or stored result name.
            column: Column to aggregate.
            operation: Aggregation operation
                ('sum', 'mean', 'min', 'max', 'count').
            output_result_name: Name to store the result under.
            group_by: Optional column(s) to group by.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - input_name (str): The input name.
                - output_result_name (str): The output result name.
                - operation (str): Aggregation operation.
                - group_by (str/list): Columns grouped by.
                - output_column_name (str): Name of the output column.
                - message (str): Error message if status is 'error'.
        """
        try:
            self.__validate_result_name(output_result_name)

            try:
                operation_enum = AggregateOperation(operation)
            except ValueError:
                return {
                    "status": "error",
                    "message": (
                        f"Invalid operation: '{operation}'. Must be one of "
                        f"{[e.value for e in AggregateOperation]}."
                    ),
                }

            self.__query_plans[output_result_name] = QueryPlan(
                sources=[input_name],
                operation=OperationType.AGGREGATE,
                aggregate_params=AggregateParams(
                    column=column,
                    operation=operation_enum,
                    group_by=group_by,
                ),
            )

            output_column_name = f"{operation}_{column}"

            return {
                "status": "ok",
                "input_name": input_name,
                "output_result_name": output_result_name,
                "operation": operation,
                "column": column,
                "group_by": group_by,
                "output_column_name": output_column_name,
            }

        except Exception as e:
            logger.debug(f"Error aggregating column: {e}")
            return {"status": "error", "message": str(e)}

    def join_tables(
        self,
        left_input_name: str,
        right_input_name: str,
        on: Union[str, List[str]],
        output_result_name: str,
        how: str = "inner",
    ) -> Dict[str, Any]:
        """Join two tables or results.

        Args:
            left_input_name: Left table or result name.
            right_input_name: Right table or result name.
            on: Column(s) to join on.
            output_result_name: Name to store the joined result under.
            how: Join type ('inner', 'left', 'right', 'full').

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - left_input_name (str): Left input name.
                - right_input_name (str): Right input name.
                - output_result_name (str): The output result name.
                - join_type (str): Type of join performed.
                - message (str): Error message if status is 'error'.
        """
        try:
            self.__validate_result_name(output_result_name)

            try:
                how_enum = JoinType(how)
            except ValueError:
                return {
                    "status": "error",
                    "message": (
                        f"Invalid join type: '{how}'. Must be one of "
                        f"{[e.value for e in JoinType]}."
                    ),
                }

            self.__query_plans[output_result_name] = QueryPlan(
                sources=[left_input_name, right_input_name],
                operation=OperationType.JOIN,
                join_params=JoinParams(on=on, how=how_enum),
            )

            return {
                "status": "ok",
                "left_input_name": left_input_name,
                "right_input_name": right_input_name,
                "output_result_name": output_result_name,
                "join_type": how,
            }

        except Exception as e:
            logger.debug(f"Error joining tables: {e}")
            return {"status": "error", "message": str(e)}

    def insert_row(self, table_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a single row into a table.

        Args:
            table_name: Table name in 'schema.table' or 'table' format.
            row: Dictionary mapping column names to values.

        Returns:
            dict: JSON with keys:
                - status (str): 'ok' or 'error'.
                - table_name (str): The table name.
                - row (dict): The inserted row data.
                - message (str): Error message if status is 'error'.
        """
        if self.__engine is None:
            return {
                "status": "error",
                "message": "SQL database engine is not configured.",
            }
        try:
            table = self.__get_table(table_name)
            with self.__engine.connect() as conn:
                conn.execute(table.insert().values(**row))
                conn.commit()
            return {
                "status": "ok",
                "table_name": table_name,
                "row": row,
            }
        except Exception as e:
            logger.debug(f"Error inserting row: {e}")
            return {"status": "error", "message": str(e)}


class SQLDatabaseToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        sql_database_tools_source_configuration: SQLDatabaseToolsSourceConfiguration = (
            SQLDatabaseToolsSourceConfiguration()
        ),
    ) -> SQLDatabaseToolsSource:
        config = sql_database_tools_source_configuration
        engine = None
        if config.enabled and config.connection_string:
            engine = _create_sql_engine(config)
        return SQLDatabaseToolsSource(config=config, engine=engine)
