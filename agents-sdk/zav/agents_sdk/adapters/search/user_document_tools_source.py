from typing import Any, Dict, List, Literal, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.retrievers.zav_retriever import ZAVRetriever
from zav.agents_sdk.adapters.search.citation import CitationConfig, CitationStore
from zav.agents_sdk.adapters.tools.document_tools import (
    _DEFAULT_DESCRIPTION_FIELD,
    _DEFAULT_METADATA_FIELDS,
    extract_hit_metadata,
)
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.adapters.user_documents.user_documents_service import (
    UserDocumentsService,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable


class UserDocumentToolsConfig(BaseModel):
    fe_base_url: str = "https://search.zeta-alpha.com"
    page_size: int = 10
    list_page_size: int = 20
    metadata_fields: List[str] = Field(
        default_factory=lambda: list(_DEFAULT_METADATA_FIELDS)
    )
    description_field: str = _DEFAULT_DESCRIPTION_FIELD
    default_retrieval_method: Literal["knn", "keyword", "mixed"] = "mixed"


class UserDocumentToolsSource(ToolsSource):

    source_name = "user_document_tools"

    def __init__(
        self,
        retriever: ZAVRetriever,
        user_documents_service: UserDocumentsService,
        user_document_tools_config: UserDocumentToolsConfig,
        citation_config: CitationConfig,
        citation_store: CitationStore,
    ):
        self.__retriever = retriever
        self.__user_documents_service = user_documents_service
        self.__config = user_document_tools_config
        self.__citation_config = citation_config
        self.__citation_store = citation_store

    async def get_tools(self) -> List[Tool]:
        return [
            Tool.from_callable(
                name="search_my_documents",
                executable=self.search_my_documents,
            ),
            Tool.from_callable(
                name="list_my_documents",
                executable=self.list_my_documents,
            ),
        ]

    @streamable(
        running_text="Searching your documents: {query}...",
        completed_text="Search complete.",
        params_transform=hide,
        response_transform=hide,
    )
    async def search_my_documents(
        self,
        query: str,
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search your private uploaded documents by content relevance.

        Args:
            query: The search query in natural language.
            page: Result page number (default: 1).
            page_size: Number of results per page.

        Returns:
            Dict with 'results' (list of document hits with metadata and
            source_url), 'total_hits', and 'page'.
        """
        effective_page_size = page_size or self.__config.page_size

        try:
            response = await self.__retriever.search(
                query_string=query,
                retrieval_unit="document",
                retrieval_method=self.__config.default_retrieval_method,
                visibility=["own_private"],
                page=page,
                page_size=effective_page_size,
            )
        except Exception as e:
            logger.exception(f"Error searching user documents: {e}")
            raise Exception(f"Could not search your documents: {e}") from e

        results = []
        for hit in response.get("hits", []):
            data = extract_hit_metadata(
                hit,
                self.__config.metadata_fields,
                self.__config.description_field,
            )
            data["document_id"] = hit.get("id", "")
            source_url = f"{self.__config.fe_base_url}{hit['document_url']}"
            data["source_url"] = source_url

            short_id = None
            if self.__citation_config.strategy == "short_id":
                doc_id = hit.get("id", "")
                hash_part = doc_id.rsplit("_", 1)[0] if "_" in doc_id else doc_id
                short_id = hash_part[-8:]
                data["short_id"] = short_id

            self.__citation_store.register_hit(
                source_url=source_url,
                context_hit=data,
                search_hit=hit,
                short_id=short_id,
            )

            results.append(data)

        return {
            "results": results,
            "total_hits": response.get("total_hits", 0),
            "page": page,
        }

    @streamable(
        running_text="Listing your documents (page {page})...",
        completed_text="Documents listed.",
        response_transform=hide,
    )
    async def list_my_documents(
        self,
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """List your uploaded documents with pagination.

        Args:
            page: Page number (default: 1).
            page_size: Number of documents per page.

        Returns:
            Dict with 'documents' (list of document metadata), 'count',
            'page', and 'page_size'.
        """
        effective_page_size = page_size or self.__config.list_page_size

        try:
            response = await self.__user_documents_service.list_documents(
                page=page,
                page_size=effective_page_size,
            )
        except Exception as e:
            logger.exception(f"Error listing user documents: {e}")
            raise Exception(f"Could not list your documents: {e}") from e

        documents = []
        for doc in response.documents:
            doc_data: Dict[str, Any] = {
                "document_id": doc.id,
                "status": doc.status.value if doc.status else None,
            }
            if doc.metadata.title:
                doc_data["title"] = doc.metadata.title
            if doc.content_file_name:
                doc_data["file_name"] = doc.content_file_name
            if doc.content_type:
                doc_data["content_type"] = doc.content_type
            if doc.metadata.source:
                doc_data["source"] = doc.metadata.source
            if doc.metadata.description:
                doc_data["description"] = doc.metadata.description
            documents.append(doc_data)

        return {
            "documents": documents,
            "count": response.count,
            "page": response.page,
            "page_size": response.page_size,
        }


class UserDocumentToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        zav_retriever: ZAVRetriever,
        user_documents_service: UserDocumentsService,
        citation_store: CitationStore,
        user_document_tools_configuration: UserDocumentToolsConfig = (
            UserDocumentToolsConfig()
        ),
        citation_configuration: CitationConfig = CitationConfig(),
    ) -> UserDocumentToolsSource:
        return UserDocumentToolsSource(
            retriever=zav_retriever,
            user_documents_service=user_documents_service,
            user_document_tools_config=user_document_tools_configuration,
            citation_config=citation_configuration,
            citation_store=citation_store,
        )
