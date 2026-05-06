import re
from typing import Any, Dict, List

from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.tools.sources.index_tools_source import IndexToolsSource
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable


class DocumentToolsSourceConfiguration(BaseModel):
    enabled: bool = False
    page_size: int = 10


class DocumentTools(ToolsSource):
    """Provides tools for interacting with individual documents: reading,
    locating by fuzzy name, and retrieving structured metadata."""

    source_name = "document_tools"

    def __init__(
        self,
        index_tools: IndexToolsSource,
        document_tools_source_configuration: DocumentToolsSourceConfiguration,
    ):
        self.__index_tools = index_tools
        self.__config = document_tools_source_configuration
        self.enabled = document_tools_source_configuration.enabled

    async def get_tools(self) -> List[Tool]:
        return [
            Tool.from_callable(
                name="read_document",
                executable=self.read_document,
            ),
            Tool.from_callable(
                name="locate_document",
                executable=self.locate_document,
            ),
            Tool.from_callable(
                name="retrieve_metadata",
                executable=self.retrieve_metadata,
            ),
        ]

    @streamable(
        running_text="Reading document...",
        completed_text="Read document.",
        params_transform=hide,
        response_transform=hide,
    )
    async def read_document(self, document_id: str) -> Dict[str, Any]:
        """Read the full content of a document by its document ID.

        Args:
            document_id: The ID of the document to read.

        Returns:
            Dict[str, Any]: A dictionary containing the document content.
        """
        return await self.__index_tools.get_document_content(document_id)

    @staticmethod
    def _normalize_locator(locator: str) -> str:
        locator = re.sub(
            r"\.(pdf|docx?|xlsx?|pptx?|txt|csv)$", "", locator, flags=re.IGNORECASE
        )
        locator = re.sub(r"[-_]+", " ", locator)
        locator = re.sub(r"\s+", " ", locator).strip()
        return locator

    @streamable(
        running_text="Searching for document: {{ locator }}...",
        completed_text="Found {{ result_count }} document{{ 's' if result_count != 1 else '' }} about {{ locator }}.",  # noqa: E501
        params_transform=hide,
        response_transform=hide,
    )
    async def locate_document(self, locator: str) -> Dict[str, Any]:
        """Locate a document by a **fuzzy** title, URL, or file name. Use this
        when the user refers to a document by name, even inexactly.

        Args:
            locator: The title, URL, or file name as mentioned by the user.

        Returns:
            Dict with 'results' (list of document metadata dicts) and 'result_count'.
        """
        exact_results = await self.__index_tools.search_documents_by_keyword(
            locator, page_size=self.__config.page_size
        )
        normalized = self._normalize_locator(locator)

        if normalized != locator:
            normalized_results = await self.__index_tools.search_documents_by_keyword(
                normalized, page_size=self.__config.page_size
            )
            seen_doc_ids: set[str] = set()
            merged: List[Dict[str, Any]] = []
            for result in exact_results + normalized_results:
                doc_id = result.get("source_url")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    merged.append(result)
            return {"results": merged, "result_count": len(merged)}

        return {"results": exact_results, "result_count": len(exact_results)}

    @streamable(
        running_text="Retrieving document metadata...",
        completed_text="Retrieved metadata of {{ count }} document{{ 's' if count != 1 else '' }}.",  # noqa: E501
        params_transform=hide,
        response_transform=hide,
    )
    async def retrieve_metadata(
        self,
        document_ids: List[str],
    ) -> Dict[str, Any]:
        """Retrieve structured metadata for a list of documents by their IDs.

        Args:
            document_ids: List of document IDs to retrieve metadata for.

        Returns:
            Dict with 'documents' (list of metadata dicts keyed by document_id)
            and 'count'.
        """
        if not document_ids:
            return {"documents": [], "count": 0}

        documents = await self.__index_tools.list_documents(
            property_name="id",
            property_values=document_ids,
        )
        return {"documents": documents, "count": len(documents)}


class DocumentToolsFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        index_tools: IndexToolsSource,
        document_tools_source_configuration: DocumentToolsSourceConfiguration = (
            DocumentToolsSourceConfiguration()
        ),
    ) -> DocumentTools:
        return DocumentTools(
            index_tools=index_tools,
            document_tools_source_configuration=document_tools_source_configuration,
        )
