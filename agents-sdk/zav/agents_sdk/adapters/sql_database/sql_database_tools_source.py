from typing import List, Optional

from zav.agents_sdk.adapters.sql_database.sql_database_tools import SQLDatabaseTools
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool


class SQLDatabaseToolsSource(ToolsSource):
    source_name = "sql_database_tools"

    def __init__(self, tools: Optional[SQLDatabaseTools] = None):
        self.__tools = tools

    async def get_tools(self) -> List[Tool]:
        if self.__tools is None:
            return []

        return [
            Tool.from_callable(
                name="sql_list_tables", executable=self.__tools.list_tables
            ),
            Tool.from_callable(
                name="sql_get_table_schema", executable=self.__tools.get_table_schema
            ),
            Tool.from_callable(
                name="sql_preview_data", executable=self.__tools.preview_data
            ),
            Tool.from_callable(name="sql_read_data", executable=self.__tools.read_data),
            Tool.from_callable(
                name="sql_count_rows", executable=self.__tools.count_rows
            ),
            Tool.from_callable(
                name="sql_get_row_by_pk", executable=self.__tools.get_row_by_pk
            ),
            Tool.from_callable(
                name="sql_select_columns", executable=self.__tools.select_columns
            ),
            Tool.from_callable(
                name="sql_get_column_values", executable=self.__tools.get_column_values
            ),
            Tool.from_callable(
                name="sql_filter_table_rows", executable=self.__tools.filter_table_rows
            ),
            Tool.from_callable(
                name="sql_sort_table_rows", executable=self.__tools.sort_table_rows
            ),
            Tool.from_callable(
                name="sql_aggregate_column", executable=self.__tools.aggregate_column
            ),
            Tool.from_callable(
                name="sql_join_tables", executable=self.__tools.join_tables
            ),
            Tool.from_callable(
                name="sql_insert_row", executable=self.__tools.insert_row
            ),
        ]


class SQLDatabaseToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls, sql_database_tools: Optional[SQLDatabaseTools] = None
    ) -> SQLDatabaseToolsSource:
        return SQLDatabaseToolsSource(tools=sql_database_tools)
