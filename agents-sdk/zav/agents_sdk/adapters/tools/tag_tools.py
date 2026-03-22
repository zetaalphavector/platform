from typing import Any, Dict, List, Union

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.search.index_tools_source import IndexToolsSource
from zav.agents_sdk.adapters.tags.tags_service import TagsService
from zav.agents_sdk.adapters.tools.document_tools import DocumentTools
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable


class TagToolsConfig(BaseModel):
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
        document_tools: DocumentTools,
        index_tools: IndexToolsSource,
        tag_tools_config: TagToolsConfig,
    ):
        self.__tags_service = tags_service
        self.__document_tools = document_tools
        self.__index_tools = index_tools
        self.__config = tag_tools_config

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
        ]

    @streamable(
        running_text="Listing tags...",
        completed_text="Loaded tags.",
        response_transform=hide,
    )
    async def list_user_tags(self) -> Dict[str, Any]:
        """List all tags that the user has access to (created by them or shared).

        Returns:
            Dict with 'results' (list of tag metadata dicts) and 'result_count'.
        """
        try:
            tags = await self.__tags_service.get_tags()
            return {"results": tags, "result_count": len(tags)}
        except Exception as e:
            logger.exception(f"Error during tag listing: {e}")
            raise Exception(f"Could not list tags: {e}") from e

    @streamable(
        running_text="Listing documents in tag (page {page})...",
        completed_text="Loaded tag documents.",
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
            "Found {{ results|length }} result"
            "{{ 's' if results|length != 1 else '' }}."
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

    async def __enhance_tagged_documents(
        self, tagged_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance tagged documents with structured metadata from DocumentTools."""
        if not tagged_documents:
            return []

        organize_doc_ids = [d["document_id"] for d in tagged_documents]
        metadata_map = (
            await self.__document_tools.retrieve_metadata_by_organize_doc_ids(
                organize_doc_ids
            )
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
        document_tools: DocumentTools,
        index_tools: IndexToolsSource,
        tag_tools_config: TagToolsConfig = TagToolsConfig(),
    ) -> TagTools:
        return TagTools(
            tags_service=tags_service,
            document_tools=document_tools,
            index_tools=index_tools,
            tag_tools_config=tag_tools_config,
        )
