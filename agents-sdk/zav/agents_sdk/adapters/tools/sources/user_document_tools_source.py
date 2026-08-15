import base64
from typing import Any, Dict, List, Optional

from zav.api.errors import UnknownException
from zav.logging import logger
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.agent_state.citation import CitationStore
from zav.agents_sdk.adapters.tools.sources.index_tools_source import IndexToolsSource
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.adapters.user_documents.user_documents_service import (
    UserDocumentCreateRequest,
    UserDocumentMetadata,
    UserDocumentsService,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable


class UserDocumentToolsSourceConfiguration(BaseModel):
    enabled: bool = False
    list_page_size: int = 20


class UserDocumentToolsSource(ToolsSource):

    source_name = "user_document_tools"

    def __init__(
        self,
        index_tools: IndexToolsSource,
        user_documents_service: UserDocumentsService,
        user_document_tools_source_configuration: UserDocumentToolsSourceConfiguration,
        citation_store: CitationStore,
    ):
        self.__index_tools = index_tools
        self.__user_documents_service = user_documents_service
        self.__config = user_document_tools_source_configuration
        self.__citation_store = citation_store
        self.enabled = user_document_tools_source_configuration.enabled

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
            Tool.from_callable(
                name="create_my_document",
                executable=self.create_my_document,
            ),
            Tool.from_callable(
                name="update_my_document",
                executable=self.update_my_document,
            ),
            Tool.from_callable(
                name="delete_my_document",
                executable=self.delete_my_document,
            ),
        ]

    @streamable(
        running_text="Searching in My Documents about {{ query }}...",
        completed_text="Retrieved {{ total_hits }} result{{ 's' if total_hits != 1 else '' }} about '{{ query }}' in My Documents.",  # noqa: E501
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
        return await self.__index_tools.search_with_visibility(
            query=query,
            visibility=["own_private"],
            page=page,
            page_size=page_size,
        )

    @streamable(
        running_text="Listing page {{ page }} of My Documents...",
        completed_text="Listed {{ documents|length }} document{{ 's' if documents|length != 1 else '' }} from page {{ page }} of My Documents.",  # noqa: E501
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
                "document_id": doc.uri_hash,
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

        organize_doc_ids = [d["document_id"] for d in documents]
        metadata_map = await self.__index_tools.retrieve_metadata_by_organize_doc_ids(
            organize_doc_ids
        )
        for doc_data in documents:
            doc_preview = metadata_map.get(doc_data["document_id"], {})
            if doc_preview:
                doc_data["document_preview"] = doc_preview

        return {
            "documents": documents,
            "count": response.count,
            "page": response.page,
            "page_size": response.page_size,
        }

    @streamable(
        running_text="Adding '{{ title }}' to My Documents...",
        completed_text=(
            "Indexed '{{ title }}' into My Documents (status: {{ status }})."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def create_my_document(
        self,
        title: str,
        content: str,
        source: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create and index a new private document from text content.

        Use this to save text (a meeting summary, a pasted article, notes)
        into the user's private "My Documents" so it becomes searchable via
        the platform's retrieval pipeline. The document is private to the user.

        Args:
            title: The document title.
            content: The full text to index (markdown or plain text).
            source: Optional origin label (e.g. "Granola", "Manual").
            authors: Optional list of author names.
            year: Optional year associated with the document.

        Returns:
            Dict with 'document_id' (uri_hash), 'status', and 'title'.
        """
        base64_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
        try:
            document = await self.__user_documents_service.create_document(
                UserDocumentCreateRequest(
                    document_metadata=UserDocumentMetadata(
                        title=title,
                        source=source,
                        authors=authors,
                        year=year,
                    ),
                    content_file_name=f"{title}.md",
                    base64_content=base64_content,
                )
            )
        except Exception as e:
            logger.exception(f"Error creating user document: {e}")
            raise Exception(f"Could not create your document: {e}") from e
        return {
            "document_id": document.uri_hash,
            "status": document.status.value if document.status else None,
            "title": document.metadata.title,
        }

    @streamable(
        running_text="Updating document metadata...",
        completed_text="Document updated.",
        params_transform=hide,
        response_transform=hide,
    )
    async def update_my_document(
        self,
        document_id: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update a private document's metadata (title, source, authors, year).

        Only the fields you pass are changed. Use list_my_documents to find the
        document_id (uri_hash).

        Args:
            document_id: The document ID (uri_hash) to update.
            title: New title, if changing.
            source: New source label, if changing.
            authors: New list of author names, if changing.
            year: New year, if changing.

        Returns:
            Dict confirming the update.
        """
        try:
            await self.__user_documents_service.update_document(
                document_id=document_id,
                title=title,
                source=source,
                authors=authors,
                year=year,
            )
        except Exception as e:
            logger.exception(f"Error updating user document: {e}")
            raise Exception(f"Could not update your document: {e}") from e
        return {"status": "updated", "document_id": document_id}

    @streamable(
        running_text="Removing document...",
        completed_text="Document removed from My Documents.",
        params_transform=hide,
        response_transform=hide,
    )
    async def delete_my_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a private document from the user's My Documents.

        Args:
            document_id: The document ID (uri_hash) to delete.

        Returns:
            Dict confirming the deletion.
        """
        try:
            await self.__user_documents_service.delete_document(document_id)
        except Exception as e:
            logger.exception(f"Error deleting user document: {e}")
            raise Exception(f"Could not delete your document: {e}") from e
        return {"status": "deleted", "document_id": document_id}

    async def ensure_document_indexed(self, document_id: str) -> str:
        hit = self.__citation_store.get_hit_by_document_id(document_id)
        if hit is None:
            return document_id

        if hit.index_type != "federated":
            return document_id

        search_hit = hit.search_hit
        organize_doc_id = search_hit.get("organize_doc_id", document_id)
        uri = search_hit.get("uri", "")
        if not uri:
            return organize_doc_id

        metadata = search_hit.get("custom_metadata", {}).get("metadata", {})
        creators = metadata.get("creator", [])
        authors = [
            c["full_name"]
            for c in creators
            if isinstance(c, dict) and c.get("full_name")
        ] or None
        year = metadata.get("date")

        logger.info("Creating external document for federated result: %s", document_id)
        try:
            external_doc = await self.__user_documents_service.create_external_document(
                uri=uri,
                title=metadata.get("title"),
                authors=authors,
                description=metadata.get("abstract"),
                year=year,
                source=metadata.get("source"),
                search_engine=hit.index_id,
            )
            return external_doc.uri_hash
        except UnknownException:
            logger.info("External document may already exist, using organize_doc_id")
            return organize_doc_id


class UserDocumentToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        index_tools: IndexToolsSource,
        user_documents_service: UserDocumentsService,
        citation_store: CitationStore,
        user_document_tools_source_configuration: UserDocumentToolsSourceConfiguration = (  # noqa: E501
            UserDocumentToolsSourceConfiguration()
        ),
    ) -> UserDocumentToolsSource:
        return UserDocumentToolsSource(
            index_tools=index_tools,
            user_documents_service=user_documents_service,
            user_document_tools_source_configuration=(
                user_document_tools_source_configuration
            ),
            citation_store=citation_store,
        )
