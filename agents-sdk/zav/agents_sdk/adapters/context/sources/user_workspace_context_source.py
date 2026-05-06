from typing import Any, Dict, List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ResolvedContextItem,
)
from zav.agents_sdk.adapters.tags.tags_service import TagsService
from zav.agents_sdk.adapters.user_documents.user_documents_service import (
    UserDocumentsService,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ConversationContext

_MAX_TAG_NAMES = 10


class UserWorkspaceContextSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description="Enable user workspace context resolution.",
    )
    include_tags: bool = Field(
        True,
        description="Include tag summary (count and names).",
    )
    include_private_documents: bool = Field(
        True,
        description="Include private document count.",
    )


class UserWorkspaceContextSource(ContextSource):
    """Resolves a lightweight snapshot of the user's workspace.

    Fetches summary-level information (counts, names) from
    platform services so the agent knows what the user has
    built — tags, uploaded documents — without loading full
    content. Memory state is already provided by the
    MemoryProvider and should not be duplicated here.
    """

    source_name = "user_workspace"

    def __init__(
        self,
        enabled: bool,
        include_tags: bool,
        include_private_documents: bool,
        tags_service: Optional[TagsService] = None,
        user_documents_service: Optional[UserDocumentsService] = None,
    ):
        self.enabled = enabled
        self.__include_tags = include_tags
        self.__include_private_documents = include_private_documents
        self.__tags_service = tags_service
        self.__user_documents_service = user_documents_service

    def describe(self, origin: ContextOrigin) -> str:
        return (
            "A snapshot of this user's workspace on the "
            "platform — what they have organized so far."
        )

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        summary: Dict[str, Any] = {}

        if self.__include_tags and self.__tags_service:
            tags_data = await self.__fetch_tags_summary()
            if tags_data is not None:
                summary["tags"] = tags_data

        if self.__include_private_documents and self.__user_documents_service:
            docs_data = await self.__fetch_docs_summary()
            if docs_data is not None:
                summary["private_documents"] = docs_data

        if not summary:
            return None

        return [
            ResolvedContextItem(
                id="workspace_summary",
                data=summary,
                item_type="workspace",
            )
        ]

    async def __fetch_tags_summary(
        self,
    ) -> Optional[Dict[str, Any]]:
        assert self.__tags_service is not None
        try:
            tags = await self.__tags_service.get_tags(page_size=_MAX_TAG_NAMES + 1)
            owned_tags = [
                t for t in tags if str(t.get("tag_type", "")).upper() != "SHARED"
            ]
            count = len(owned_tags)
            names = [
                t.get("name", "") for t in owned_tags[:_MAX_TAG_NAMES] if t.get("name")
            ]
            result: Dict[str, Any] = {
                "count": count,
                "names": names,
            }
            if count > _MAX_TAG_NAMES:
                result["truncated"] = True
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch tags summary for workspace context: {e}")
            return None

    async def __fetch_docs_summary(
        self,
    ) -> Optional[Dict[str, Any]]:
        assert self.__user_documents_service is not None
        try:
            response = await self.__user_documents_service.list_documents(
                page=1, page_size=1
            )
            return {"count": response.count}
        except Exception as e:
            logger.warning(
                "Failed to fetch private documents summary "
                f"for workspace context: {e}"
            )
            return None


class UserWorkspaceContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        user_workspace_context_source_configuration: (
            UserWorkspaceContextSourceConfiguration
        ) = UserWorkspaceContextSourceConfiguration(),
        tags_service: Optional[TagsService] = None,
        user_documents_service: Optional[UserDocumentsService] = None,
    ) -> UserWorkspaceContextSource:
        cfg = user_workspace_context_source_configuration
        return UserWorkspaceContextSource(
            enabled=cfg.enabled,
            include_tags=cfg.include_tags,
            include_private_documents=(cfg.include_private_documents),
            tags_service=tags_service,
            user_documents_service=user_documents_service,
        )
