import json
from typing import Any, Dict, List, Optional

from zav.agents_sdk.adapters.context.context_source import ResolvedContextItem


def format_single_document_xml(
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


def format_document_items_xml(
    items: List[ResolvedContextItem],
    max_item_content_length: Optional[int] = None,
) -> str:
    """Format document items as XML, truncating descriptions to fit within limits."""
    return "\n".join(
        format_single_document_xml(
            document_id=item.data.get("document_id", item.id),
            doc_data=item.data,
            max_content_length=max_item_content_length,
            extra_metadata={
                key: item.data[key]
                for key in ("source_url", "short_id")
                if item.data.get(key)
            },
        )
        for item in items
    )
