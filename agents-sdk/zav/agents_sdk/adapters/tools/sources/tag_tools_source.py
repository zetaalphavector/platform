from typing import Any, Dict, List, Optional, Union

from zav.api.errors import UnknownException
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.tags.tags_service import TagsService
from zav.agents_sdk.adapters.tools.sources.index_tools_source import IndexToolsSource
from zav.agents_sdk.adapters.tools.sources.user_document_tools_source import (
    UserDocumentToolsSource,
)
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable


class TagToolsSourceConfiguration(BaseModel):
    enabled: bool = False
    max_documents_per_page: int = Field(
        20,
        description=(
            "Maximum number of documents to return per"
            " page when listing tag contents."
        ),
    )


class TagTools(ToolsSource):
    """Provides tools for interacting with user tags: listing tags and
    browsing documents within a tag."""

    source_name = "tag_tools"

    def __init__(
        self,
        tags_service: TagsService,
        index_tools: IndexToolsSource,
        tag_tools_source_configuration: TagToolsSourceConfiguration,
        user_document_tools: UserDocumentToolsSource,
    ):
        self.__tags_service = tags_service
        self.__index_tools = index_tools
        self.__config = tag_tools_source_configuration
        self.__user_document_tools = user_document_tools
        self.enabled = tag_tools_source_configuration.enabled

    async def get_tools(self) -> List[Tool]:
        return [
            Tool.from_callable(
                name="list_user_tags",
                executable=self.list_user_tags,
            ),
            Tool.from_callable(
                name="list_documents_in_tag",
                executable=self.list_documents_in_tag,
                description=(
                    "List documents in a tag with pagination."
                    f" Each page shows up to"
                    f" {self.__config.max_documents_per_page} documents."
                    "\n\nArgs:\n"
                    "    tag_id: The ID of the tag to list documents from.\n"
                    "    page: The page number to retrieve (default: 1).\n\n"
                    "Returns:\n"
                    "    Dict containing tag metadata, documents with"
                    " structured metadata,\n"
                    "    pagination info (total_documents, page,"
                    " total_pages, has_more_documents)."
                ),
            ),
            Tool.from_callable(
                name="search_in_tag",
                executable=self.search_in_tag,
            ),
            Tool.from_callable(
                name="tag_document",
                executable=self.tag_document,
            ),
            Tool.from_callable(
                name="untag_document",
                executable=self.untag_document,
            ),
            Tool.from_callable(
                name="create_tag",
                executable=self.create_tag,
            ),
        ]

    @streamable(
        running_text="Listing tags...",
        completed_text=(
            "Found {{ result_count }}" " tag{{ 's' if result_count != 1 else '' }}."
        ),
        response_transform=hide,
    )
    async def list_user_tags(self) -> Dict[str, Any]:
        """List all tags that the user has access to (created by them or shared).

        Returns:
            Dict with 'results' (list of tag metadata dicts) and 'result_count'.
        """
        try:
            tags = await self.__tags_service.get_tags()
            return {
                "results": [
                    {
                        "id": tag.get("id"),
                        "name": tag.get("name"),
                        "description": tag.get("description"),
                        "tag_type": tag.get("tag_type"),
                        "stats": tag.get("stats"),
                        "date_created": tag.get("date_created"),
                        "date_modified": tag.get("date_modified"),
                        "permission": tag.get("permission"),
                        "settings": tag.get("settings"),
                    }
                    for tag in tags
                ],
                "result_count": len(tags),
            }
        except Exception as e:
            logger.exception(f"Error during tag listing: {e}")
            raise Exception(f"Could not list tags: {e}") from e

    @streamable(
        running_text="Listing documents in tag (page {{ page }})...",
        completed_text="Loaded documents of tag '{{ tag_name }}' (page {{ page }}).",
        params_transform=hide,
        response_transform=hide,
    )
    async def list_documents_in_tag(
        self,
        tag_id: int,
        page: int = 1,
    ) -> Dict[str, Any]:
        """List documents in a tag with pagination.

        Args:
            tag_id: The ID of the tag to list documents from.
            page: The page number to retrieve (default: 1).

        Returns:
            Dict containing tag metadata, documents with structured metadata,
            pagination info (total_documents, page, total_pages, has_more_documents).
        """
        try:
            tag_metadata = await self.__tags_service.get_tag_by_id(tag_id)
            if not tag_metadata:
                raise ValueError(f"Tag with ID {tag_id} not found")

            page_size = self.__config.max_documents_per_page
            tag_response = await self.__tags_service.get_tagged_documents_paginated(
                tag_id=tag_id,
                tag_type=tag_metadata.get("tag_type", "own"),
                page=page,
                page_size=page_size,
            )

            tagged_documents = tag_response["results"]
            total_documents = tag_response["total_count"]
            total_pages = (total_documents + page_size - 1) // page_size

            if not tagged_documents:
                return {
                    "tag_id": str(tag_metadata["id"]),
                    "tag_name": tag_metadata.get("name"),
                    "tag_date_created": tag_metadata.get("date_created"),
                    "tag_description": tag_metadata.get("description"),
                    "total_documents": total_documents,
                    "tag_chip_color": tag_metadata.get("color"),
                    "tag_sharing_policy": tag_metadata.get("sharing"),
                    "tagged_documents": {
                        "paginated_documents": [],
                        "total_pages": total_pages,
                        "has_more_documents": False,
                    },
                }

            enhanced_documents = await self.__enhance_tagged_documents(tagged_documents)

            return {
                "tag_id": str(tag_metadata["id"]),
                "tag_name": tag_metadata.get("name"),
                "tag_date_created": tag_metadata.get("date_created"),
                "tag_description": tag_metadata.get("description"),
                "total_documents": total_documents,
                "tag_chip_color": tag_metadata.get("color"),
                "tag_sharing_policy": tag_metadata.get("sharing"),
                "tagged_documents": {
                    "paginated_documents": [
                        {
                            "documents": enhanced_documents,
                            "page": page,
                        }
                    ],
                    "total_pages": total_pages,
                    "has_more_documents": page < total_pages,
                },
            }
        except Exception as e:
            logger.exception(f"Error during tag document listing: {e}")
            raise Exception(f"Could not list documents in tag: {e}") from e

    @streamable(
        running_text="Searching in tag for: {{ query }}...",
        completed_text=(
            "Retrieved {{ results|length }} result"
            "{{ 's' if results|length != 1 else '' }} about '{{ query }}' from tag."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def search_in_tag(
        self,
        tag_id: Union[int, str],
        query: str,
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        """Search for documents within a specific tag.

        Runs a semantic + keyword search scoped to the documents in the
        given tag. Use list_user_tags first to discover available tags
        and their IDs.

        Args:
            tag_id: The ID of the tag to search within.
            query: Search query (natural language or keyword).
            page: Result page number (default: 1).
            page_size: Number of results per page (default: 10).

        Returns:
            Dict with 'results' (list of document hits with metadata),
            'total_hits', and 'page'.
        """
        return await self.__index_tools.search_with_tag_ids(
            query=query,
            tag_ids=[str(tag_id)],
            page=page,
            page_size=page_size,
        )

    @streamable(
        running_text="Adding document to tag...",
        completed_text="Document added to {{ tag_name }}.",
        params_transform=hide,
        response_transform=hide,
    )
    async def tag_document(
        self,
        tag_id: int,
        document_id: str,
    ) -> Dict[str, Any]:
        """Add a document to a tag.

        Args:
            tag_id: The ID of the tag to add the document to.
            document_id: The document ID (uri_hash) to add.

        Returns:
            Dict confirming the document was added.
        """
        try:
            tag = await self.__tags_service.get_tag_by_id(tag_id)
            if not tag:
                raise ValueError(f"Tag with ID {tag_id} not found")
            tag_type = tag.get("tag_type", "own").lower()
            tag_name = tag.get("name", str(tag_id))

            document_id = await self.__user_document_tools.ensure_document_indexed(
                document_id
            )

            result = await self.__tags_service.tag_document(
                tag_id=tag_id,
                tag_type=tag_type,
                uri_hash=document_id,
            )
            return {
                "status": "added",
                "tag_id": tag_id,
                "tag_name": tag_name,
                "document_id": document_id,
                **result,
            }
        except Exception as e:
            if isinstance(e, UnknownException) and "already tagged" in str(e).lower():
                return {
                    "status": "already_tagged",
                    "tag_id": tag_id,
                    "tag_name": tag_name,
                    "document_id": document_id,
                }
            logger.exception(f"Error adding document to tag: {e}")
            raise Exception(f"Could not add document to tag: {e}") from e

    @streamable(
        running_text="Removing document from tag...",
        completed_text="Document removed from {{ tag_name }}.",
        params_transform=hide,
        response_transform=hide,
    )
    async def untag_document(
        self,
        tag_id: int,
        document_id: str,
    ) -> Dict[str, Any]:
        """Remove a document from a tag.

        Args:
            tag_id: The ID of the tag to remove the document from.
            document_id: The document ID (uri_hash) to remove.

        Returns:
            Dict confirming the document was removed.
        """
        try:
            tag = await self.__tags_service.get_tag_by_id(tag_id)
            if not tag:
                raise ValueError(f"Tag with ID {tag_id} not found")
            tag_type = tag.get("tag_type", "own").lower()
            tag_name = tag.get("name", str(tag_id))
            await self.__tags_service.untag_document(
                tag_id=tag_id,
                tag_type=tag_type,
                uri_hash=document_id,
            )
            return {
                "status": "removed",
                "tag_id": tag_id,
                "tag_name": tag_name,
                "document_id": document_id,
            }
        except Exception as e:
            logger.exception(f"Error removing document from tag: {e}")
            raise Exception(f"Could not remove document from tag: {e}") from e

    @streamable(
        running_text="Creating tag {{ name }}...",
        completed_text="Tag {{ name }} created.",
        params_transform=hide,
        response_transform=hide,
    )
    async def create_tag(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new personal tag for the user.

        Args:
            name: The name for the new tag.
            description: Optional description of the tag's purpose.

        Returns:
            Dict with the created tag metadata including its ID.
        """
        try:
            result = await self.__tags_service.create_tag(
                name=name,
                description=description,
                recommendations_enabled=True,
            )
            return {"status": "created", **result}
        except Exception as e:
            logger.exception(f"Error creating tag: {e}")
            raise Exception(f"Could not create tag: {e}") from e

    async def __enhance_tagged_documents(
        self, tagged_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance tagged documents with structured metadata from DocumentTools."""
        if not tagged_documents:
            return []

        organize_doc_ids = [d["document_id"] for d in tagged_documents]
        metadata_map = await self.__index_tools.retrieve_metadata_by_organize_doc_ids(
            organize_doc_ids
        )

        enhanced_docs = []
        for tagged_doc in tagged_documents:
            doc_id = tagged_doc["document_id"]
            enhanced_doc: Dict[str, Any] = {
                "document_id": doc_id,
                "date_tagged": tagged_doc.get("date_tagged"),
                "permission": tagged_doc.get("permission"),
                "user_order": tagged_doc.get("user_order"),
            }
            doc_metadata = metadata_map.get(doc_id, {})
            if doc_metadata:
                enhanced_doc["document_preview"] = doc_metadata
            enhanced_docs.append(enhanced_doc)

        return enhanced_docs


class TagToolsFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        tags_service: TagsService,
        index_tools: IndexToolsSource,
        user_document_tools: UserDocumentToolsSource,
        tag_tools_source_configuration: TagToolsSourceConfiguration = (
            TagToolsSourceConfiguration()
        ),
    ) -> TagTools:
        return TagTools(
            tags_service=tags_service,
            index_tools=index_tools,
            tag_tools_source_configuration=tag_tools_source_configuration,
            user_document_tools=user_document_tools,
        )
