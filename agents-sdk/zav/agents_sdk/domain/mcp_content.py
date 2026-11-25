import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

FILE_NOT_PASSED_NOTE = "The file's content is not passed on."


def _mime_type(fields: Dict[str, Any]) -> Optional[str]:
    # The MCP SDK dumps its models as `mime_type`; the wire name is `mimeType`.
    return fields.get("mime_type") or fields.get("mimeType")


@dataclass(frozen=True)
class FileContent:
    # Base64, as MCP carries it.
    data: str
    mime_type: Optional[str]
    uri: Optional[str]


def file_content(part: Any) -> Optional[FileContent]:
    """The bytes an MCP content part carries: an embedded resource's `blob`, or
    an image's or audio's `data`. None for any other part."""
    if not isinstance(part, dict):
        return None
    resource = part.get("resource")
    if (
        part.get("type") == "resource"
        and isinstance(resource, dict)
        and isinstance(resource.get("blob"), str)
    ):
        return FileContent(resource["blob"], _mime_type(resource), resource.get("uri"))
    if part.get("type") in ("image", "audio") and isinstance(part.get("data"), str):
        return FileContent(part["data"], _mime_type(part), None)
    return None


def _parts_key(response: Dict[str, Any]) -> Optional[str]:
    # No MCP part has `content`: a dict with a `content` list is a CallToolResult.
    return next(
        (key for key in ("parts", "content") if isinstance(response.get(key), list)),
        None,
    )


def replace_files(
    response: Dict[str, Any], replaced: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """``response`` with the parts at the given positions replaced; a response
    that is a single part is position 0."""
    key = _parts_key(response)
    if key is None:
        return replaced.get(0, response)
    return {
        **response,
        key: [replaced.get(i, part) for i, part in enumerate(response[key])],
    }


def file_positions(response: Optional[Dict[str, Any]]) -> Dict[int, FileContent]:
    """The parts of an MCP tool result, as the MCP tools provider returns it
    (one part, ``{"parts": [...]}``, or the CallToolResult of a tool that views
    use), that carry bytes, by position."""
    if not isinstance(response, dict):
        return {}
    key = _parts_key(response)
    candidates = [response] if key is None else response[key]
    found = {i: file_content(part) for i, part in enumerate(candidates)}
    return {i: content for i, content in found.items() if content is not None}


def without_files(response: Optional[Dict]) -> Optional[Dict]:
    """The result with each part's bytes replaced by a note: base64 is of no use
    to the model or the user."""
    files = file_positions(response)
    if not files:
        return response
    notes = {
        i: {
            "type": "text",
            "text": json.dumps(
                {
                    "file": content.uri,
                    "content_type": content.mime_type,
                    "note": FILE_NOT_PASSED_NOTE,
                }
            ),
        }
        for i, content in files.items()
    }
    return replace_files(cast(Dict[str, Any], response), notes)
