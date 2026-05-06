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
from zav.agents_sdk.adapters.tools.sources.tag_tools_source import TagTools
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ConversationContext


class TagContextSourceConfiguration(BaseModel):
    enabled: bool = Field(True, description="Enable tag context resolution.")


class TagContextSource(ContextSource):
    """Resolves tag context by fetching tag metadata and document listings."""

    source_name = "tags"

    def __init__(
        self,
        tag_tools: TagTools,
        enabled: bool,
    ):
        self.enabled = enabled
        self.__tag_tools = tag_tools

    def describe(self, origin: ContextOrigin) -> str:
        if origin == ContextOrigin.INITIAL:
            return (
                "The user is currently in their tags page, working within these tags:"
            )
        return "The user added these tags as context in this message:"

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        tag_ctx = context.tag_context
        if not tag_ctx or not tag_ctx.tag_ids:
            return None

        items: List[ResolvedContextItem] = []
        for tag_id in tag_ctx.tag_ids:
            try:
                tag_info = await self.__tag_tools.list_documents_in_tag(
                    tag_id=int(tag_id), page=1
                )
                items.append(
                    ResolvedContextItem(id=tag_id, data=tag_info, item_type="tag")
                )
            except Exception as e:
                logger.warning(f"Failed to resolve tag {tag_id}: {e}")
                items.append(
                    ResolvedContextItem(
                        id=tag_id, data={"tag_id": tag_id}, item_type="tag"
                    )
                )

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

        parts: List[str] = []
        for item in items:
            tag_data = item.data
            tag_meta = {
                k: v
                for k, v in tag_data.items()
                if k not in ("tag_id", "tagged_documents", "total_documents")
            }
            tag_meta_json = json.dumps(
                tag_meta, indent=2, default=str, ensure_ascii=False
            )

            is_summarized = (
                max_item_content_length is not None
                and len(tag_meta_json) > max_item_content_length
            )
            tag_id = tag_data.get("tag_id", item.id)
            if is_summarized:
                parts.append(f'<item tag_id="{tag_id}" isSummarized="true"/>')
                continue

            parts.append(f'<item tag_id="{tag_id}">')
            parts.append(tag_meta_json)

            tagged_docs = tag_data.get("tagged_documents", {})
            total_documents = tag_data.get("total_documents", 0)
            total_pages = tagged_docs.get("total_pages", 1)
            has_more = tagged_docs.get("has_more_documents", False)

            doc_list: List[Dict[str, Any]] = []
            for page_data in tagged_docs.get("paginated_documents", []):
                doc_list.extend(page_data.get("documents", []))

            if doc_list:
                parts.append(
                    f"The following are the top {len(doc_list)} documents"
                    f" in this tag (out of {total_documents} total)."
                )
                if has_more:
                    remaining = total_documents - len(doc_list)
                    parts.append(
                        f"There are {remaining} more documents across"
                        f" {total_pages - 1} additional pages."
                    )

            for doc in doc_list:
                doc_id = doc.get("document_id", "unknown")
                doc_preview = doc.get("document_preview", {})
                tag_meta = {
                    k: v
                    for k, v in doc.items()
                    if k not in ("document_id", "document_preview") and v is not None
                }
                parts.append(
                    format_single_document_xml(
                        document_id=doc_id,
                        doc_data=doc_preview,
                        max_content_length=max_item_content_length,
                        extra_metadata=tag_meta,
                    )
                )

            parts.append("</item>")
        return "\n".join(parts)


class TagContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        tag_tools: TagTools,
        tag_context_source_configuration: TagContextSourceConfiguration = (
            TagContextSourceConfiguration()
        ),
    ) -> TagContextSource:
        return TagContextSource(
            tag_tools=tag_tools,
            enabled=tag_context_source_configuration.enabled,
        )
