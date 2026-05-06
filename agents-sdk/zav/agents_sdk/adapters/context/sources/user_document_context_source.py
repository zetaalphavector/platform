import json
from typing import Any, Dict, List, Literal, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ResolvedContextItem,
)
from zav.agents_sdk.adapters.context.sources._formatting import (
    format_single_document_xml,
)
from zav.agents_sdk.adapters.tools.sources.user_document_tools_source import (
    UserDocumentToolsSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ConversationContext


class UserDocumentContextSourceConfiguration(BaseModel):
    enabled: bool = Field(True, description="Enable user document context resolution.")


class UserDocumentContextSource(ContextSource):
    """Resolves user document context by listing uploaded documents."""

    source_name = "user_documents"

    def __init__(
        self,
        user_document_tools: UserDocumentToolsSource,
        enabled: bool,
    ):
        self.enabled = enabled
        self.__user_document_tools = user_document_tools

    def describe(self, origin: ContextOrigin) -> str:
        if origin == ContextOrigin.INITIAL:
            return (
                "The user is currently in their My Documents page, "
                "viewing their uploaded documents:"
            )
        return "The user added their uploaded documents as context in this message:"

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        user_doc_ctx = context.user_document_context
        if not user_doc_ctx or not user_doc_ctx.enabled:
            return None

        try:
            result = await self.__user_document_tools.list_my_documents(page=1)
            return [
                ResolvedContextItem(
                    id="user_documents",
                    data=result,
                    item_type="user_documents",
                )
            ]
        except Exception as e:
            logger.warning(f"Failed to resolve user document context: {e}")
            return [
                ResolvedContextItem(
                    id="user_documents",
                    data={"documents": [], "count": 0},
                    item_type="user_documents",
                )
            ]

    def format_items(
        self,
        items: List[ResolvedContextItem],
        content_format: Literal["json", "xml"],
        max_item_content_length: Optional[int] = None,
    ) -> str:
        if content_format != "xml":
            return json.dumps(
                [item.data for item in items], default=str, ensure_ascii=False
            )

        parts: List[str] = []
        for item in items:
            data = item.data
            documents = data.get("documents", [])
            count = data.get("count", 0)

            parts.append(f'<user_documents total="{count}">')

            if documents:
                parts.append(
                    f"The following are the first {len(documents)} documents"
                    f" in My Documents (out of {count} total)."
                )

            for doc in documents:
                doc_id = doc.get("document_id", "unknown")
                doc_preview = doc.get("document_preview", {})
                doc_meta: Dict[str, Any] = {
                    k: v
                    for k, v in doc.items()
                    if k not in ("document_id", "document_preview") and v is not None
                }
                parts.append(
                    format_single_document_xml(
                        document_id=doc_id,
                        doc_data=doc_preview,
                        max_content_length=max_item_content_length,
                        extra_metadata=doc_meta,
                    )
                )

            parts.append("</user_documents>")
        return "\n".join(parts)


class UserDocumentContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        user_document_tools: UserDocumentToolsSource,
        user_document_context_source_configuration: (
            UserDocumentContextSourceConfiguration
        ) = UserDocumentContextSourceConfiguration(),
    ) -> UserDocumentContextSource:
        return UserDocumentContextSource(
            user_document_tools=user_document_tools,
            enabled=user_document_context_source_configuration.enabled,
        )
