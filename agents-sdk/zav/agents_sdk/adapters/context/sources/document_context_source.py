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
    format_document_items_xml,
)
from zav.agents_sdk.adapters.tools.sources.document_tools_source import DocumentTools
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ConversationContext


class DocumentContextSourceConfiguration(BaseModel):
    enabled: bool = Field(True, description="Enable document context resolution.")


class DocumentContextSource(ContextSource):
    """Resolves document context by fetching previews from the retriever."""

    source_name = "documents"

    def __init__(
        self,
        document_tools: DocumentTools,
        enabled: bool,
    ):
        self.enabled = enabled
        self.__document_tools = document_tools

    def describe(self, origin: ContextOrigin) -> str:
        if origin == ContextOrigin.INITIAL:
            return "The user is currently viewing the following documents:"
        return "The user attached the following documents in this message:"

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        doc_ctx = context.document_context
        if not doc_ctx or not doc_ctx.document_ids:
            return None

        document_ids = doc_ctx.document_ids
        if doc_ctx.retrieval_unit == "chunk":
            document_ids = list(
                dict.fromkeys(f"{_id.split('_')[0]}_0" for _id in document_ids)
            )

        if not document_ids:
            return None

        metadata_map: Dict[str, Dict[str, Any]] = {}
        try:
            result = await self.__document_tools.retrieve_metadata(
                document_ids=document_ids
            )
            metadata_map = {
                doc["document_id"]: doc for doc in result.get("documents", [])
            }
        except Exception:
            logger.warning("Failed to retrieve document metadata for context")

        items: List[ResolvedContextItem] = []
        for did in document_ids:
            data = metadata_map.get(did, {})
            items.append(ResolvedContextItem(id=did, data=data, item_type="document"))

        return items

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
        return format_document_items_xml(items, max_item_content_length)


class DocumentContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        document_tools: DocumentTools,
        document_context_source_configuration: DocumentContextSourceConfiguration = (
            DocumentContextSourceConfiguration()
        ),
    ) -> DocumentContextSource:
        return DocumentContextSource(
            document_tools=document_tools,
            enabled=document_context_source_configuration.enabled,
        )
