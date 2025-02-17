import io
from typing import Dict, List, Literal, Optional

import pandas as pd
import tabulate

from zav.agents_sdk.domain.chat_message import ContentPart, ContentPartTable


def escape_newlines(text):
    if isinstance(text, str):
        return text.replace("\n", "<br>")
    return text


def escape_pipes(text):
    if isinstance(text, str):
        return text.replace("|", "\\|")
    return text


def escape_markdown(text):
    if isinstance(text, str):
        return escape_pipes(escape_newlines(text))
    return text


class Table:
    def __init__(
        self,
        index_column: str,
        data: Optional[List[Dict[str, str]]] = None,
    ):
        self.__indexed = False
        if data is None:
            self.__df = pd.DataFrame()
            self.__index_column = index_column
            return

        self.__index_column = index_column
        df = pd.DataFrame(data).astype(str)
        if index_column not in df.columns:
            raise ValueError(
                f"Index column '{index_column}' does not exist in the data"
            )

        try:
            df.set_index(index_column, inplace=True, verify_integrity=True)
            self.__indexed = True
        except ValueError:
            raise ValueError(
                f"Index column '{index_column}' must contain unique values"
            )

        self.__df = df

    def get_index_column(self) -> Optional[str]:
        return self.__index_column

    def get_columns(self) -> List[str]:
        return (
            list(self.__df.reset_index().columns)
            if self.__indexed
            else list(self.__df.columns)
        )

    def get_row(self, row_index: str) -> Optional[Dict[str, str]]:
        if row_index not in self.__df.index:
            return None

        row = {self.__index_column: row_index} if self.__indexed else {}
        row.update(
            {str(k): str(v) for k, v in self.__df.loc[row_index].to_dict().items()}
        )
        return row

    def add_row(self, row: Dict[str, str]) -> "Table":
        row = {k: str(v) for k, v in row.items()}

        for col in self.get_columns():
            if col != self.__index_column and col not in row:
                row[col] = ""

        if self.__index_column not in row:
            raise ValueError(f"Index column '{self.__index_column}' is missing")
        elif row[self.__index_column] in self.__df.index:
            raise ValueError(f"Duplicate index value in column '{self.__index_column}'")

        if self.__df.empty:
            self.__df = pd.DataFrame([row]).set_index(self.__index_column)
            self.__indexed = True
            return self

        row_df = pd.DataFrame([row]).set_index(self.__index_column)
        self.__df = pd.concat([self.__df, row_df])
        return self

    def add_column(
        self,
        column_name: str,
        column: Optional[List[str]] = None,
        default_value: str = "",
    ) -> "Table":
        if column_name in self.get_columns():
            raise ValueError(f"Column '{column_name}' already exists")

        if column is None:
            self.__df[column_name] = default_value
        elif len(column) == len(self.__df):
            self.__df[column_name] = column
        else:
            raise ValueError(
                f"Column '{column_name}' must have the same number "
                f"of rows ({len(self.__df)}) as the table"
            )
        return self

    def update_cell(self, row_index: str, column_name: str, value: str) -> "Table":
        if row_index not in self.__df.index:
            raise ValueError(f"Row '{row_index}' does not exist")

        if column_name not in self.get_columns():
            raise ValueError(f"Column '{column_name}' does not exist")

        self.__df.at[row_index, column_name] = value
        return self

    def to_rows(self) -> List[Dict[str, str]]:
        df_reset = self.__df.reset_index() if self.__indexed else self.__df
        return [
            {str(k): str(v) for k, v in row.items()}
            for row in df_reset.to_dict(orient="records")
        ]

    def to_columnar(self) -> Dict[str, List[str]]:
        df_reset = self.__df.reset_index() if self.__indexed else self.__df
        return {str(k): list(v) for k, v in df_reset.to_dict(orient="list").items()}

    def __transpose(self) -> "Table":
        df_reset = self.__df.reset_index() if self.__indexed else self.__df
        transposed_df = df_reset.transpose()
        if transposed_df.empty:
            return Table(data=None, index_column=self.__index_column)

        new_columns = transposed_df.iloc[0]
        transposed_df = transposed_df.iloc[1:]
        transposed_df.columns = new_columns

        data = (
            transposed_df.reset_index()
            .rename(columns={"index": self.__index_column})
            .to_dict("records")
        )
        transposed_table = Table(
            data=[{str(k): str(v) for k, v in row.items()} for row in data],
            index_column=self.__index_column,
        )
        return transposed_table

    def to_csv(self, transpose: bool = False) -> str:
        if transpose:
            return self.__transpose().to_csv()

        output = io.StringIO()
        df_reset = self.__df.reset_index() if self.__indexed else self.__df
        df_reset.to_csv(output, index=False)
        return output.getvalue().strip()

    def to_markdown(self, transpose: bool = False) -> str:
        if transpose:
            return self.__transpose().to_markdown()

        _new_df = self.__df.applymap(escape_markdown)
        _new_df.index = _new_df.index.map(escape_markdown)
        _new_df.columns = _new_df.columns.map(escape_markdown)
        return tabulate.tabulate(_new_df, headers="keys", tablefmt="github")

    def to_content_part(
        self, format: Literal["row", "columnar"] = "row"
    ) -> ContentPart:
        return (
            ContentPart(
                type="table",
                table=ContentPartTable(
                    rows=self.to_rows(),
                    headers=self.get_columns(),
                ),
            )
            if format == "row"
            else ContentPart(
                type="table",
                table=ContentPartTable(
                    columns=self.to_columnar(),
                    headers=self.get_columns(),
                ),
            )
        )
