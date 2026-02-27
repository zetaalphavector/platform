import json
from typing import Any, Dict, List, Literal, Optional

from zav.logging import logger

from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ResolvedContextItem,
)
from zav.agents_sdk.adapters.tools.document_tools import (
    DocumentTools,
    DocumentToolsConfig,
)
from zav.agents_sdk.adapters.tools.tag_tools import TagTools
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ConversationContext


class DocumentContextSource(ContextSource):
    """Resolves document context by fetching previews from the retriever."""

    source_name = "documents"

    def __init__(
        self,
        document_tools: Optional[DocumentTools] = None,
        document_tools_config: Optional[DocumentToolsConfig] = None,
    ):
        self.__document_tools = document_tools
        config = document_tools_config or DocumentToolsConfig()
        self.__metadata_fields = config.metadata_fields
        self.__description_field = config.description_field

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
            document_ids = list({f"{_id.split('_')[0]}_0" for _id in document_ids})

        if not document_ids:
            return None

        metadata_map: Dict[str, Dict[str, Any]] = {}
        if self.__document_tools:
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
            items.append(ResolvedContextItem(id=f"document:{did}", data=data))

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
        return _format_document_items_xml(items, max_item_content_length)


class DocumentContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        document_tools: Optional[DocumentTools] = None,
        document_tools_config: Optional[DocumentToolsConfig] = None,
    ) -> DocumentContextSource:
        return DocumentContextSource(
            document_tools=document_tools,
            document_tools_config=document_tools_config,
        )


def _format_single_document_xml(
    document_id: str,
    doc_data: Dict[str, Any],
    max_content_length: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Format a single document as XML with consistent structure.

    Produces XML of the form:
    <item document_id="123" content_type="application/pdf">
      { ... JSON body with metadata and description... }
      <description>...</description>

    ``extra_metadata`` fields (e.g. tag-specific date_tagged, permission,
    user_order) are merged *before* the document metadata so they appear first
    in the JSON body.
    """
    content_type = doc_data.get("content_type", "unknown")
    metadata = doc_data.get("metadata", {})
    description = doc_data.get("description", "")

    body = {**(extra_metadata or {}), **metadata}
    body_json = json.dumps(body, indent=2, default=str, ensure_ascii=False)

    open_tag = f'<item document_id="{document_id}" content_type="{content_type}">'
    close_tag = "</item>"

    if max_content_length is not None:
        fixed_len = len(open_tag) + len(body_json) + len(close_tag)
        available = max_content_length - fixed_len
        if available <= 0:
            return (
                f'<item document_id="{document_id}" content_type="{content_type}"'
                f' isSummarized="true"/>'
            )
        if description and len(description) > available:
            description = description[:available] + "..."

    parts = [open_tag, body_json]
    if description:
        parts.append(f"<description>{description}</description>")
    parts.append(close_tag)
    return "\n".join(parts)


def _format_document_items_xml(
    items: List[ResolvedContextItem],
    max_item_content_length: Optional[int] = None,
) -> str:
    """Format document items as XML, truncating descriptions to fit within limits."""
    return "\n".join(
        _format_single_document_xml(
            document_id=item.data.get("document_id", item.id),
            doc_data=item.data,
            max_content_length=max_item_content_length,
        )
        for item in items
    )


class TagContextSource(ContextSource):
    """Resolves tag context by fetching tag metadata and document listings."""

    source_name = "tags"

    def __init__(
        self,
        tag_tools: Optional[TagTools] = None,
    ):
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
        if not self.__tag_tools:
            return None

        items: List[ResolvedContextItem] = []
        for tag_id in tag_ctx.tag_ids:
            try:
                tag_info = await self.__tag_tools.list_documents_in_tag(
                    tag_id=int(tag_id), page=1
                )
                items.append(ResolvedContextItem(id=f"tag:{tag_id}", data=tag_info))
            except Exception as e:
                logger.warning(f"Failed to resolve tag {tag_id}: {e}")
                items.append(
                    ResolvedContextItem(id=f"tag:{tag_id}", data={"tag_id": tag_id})
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
                    _format_single_document_xml(
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
        tag_tools: Optional[TagTools] = None,
    ) -> TagContextSource:
        return TagContextSource(
            tag_tools=tag_tools,
        )


class CustomContextSource(ContextSource):
    """Pass-through for custom context items."""

    source_name = "custom"

    def describe(self, origin: ContextOrigin) -> str:
        if origin == ContextOrigin.INITIAL:
            return "Custom context for this conversation:"
        return "Custom context added by the user in this message:"

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        custom_ctx = context.custom_context
        if not custom_ctx or not custom_ctx.items:
            return None

        return [
            ResolvedContextItem(id=f"custom:{i}", data={"content": item.content})
            for i, item in enumerate(custom_ctx.items)
        ]


class CustomContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(cls) -> CustomContextSource:
        return CustomContextSource()
