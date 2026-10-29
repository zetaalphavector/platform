import json
from typing import Any, Dict, List, Optional, Tuple

# Streams a chat turn as Open Responses events (https://www.openresponses.org):
# the connection's first event is `response.created` with the whole message,
# then only changes. One output item per content part, in order, after a fixed
# reasoning slot at output_index 0. What the standard has no event for travels
# in `zetaalpha:`-prefixed extension events and fields, as the spec requires.
# The reasoning item is always present (empty summary until the model reasons),
# so output_index never shifts.
Event = Tuple[str, Dict[str, Any]]

SLUG = "zetaalpha"
REASONING_INDEX = 0
MESSAGE_FIELD = f"{SLUG}:message"
TOOL_FIELD = f"{SLUG}:tool"
PART_FIELD = f"{SLUG}:part"
UPDATED_ITEM_EVENT = f"{SLUG}:output_item.updated"
UPDATED_MESSAGE_EVENT = f"{SLUG}:message.updated"
SNAPSHOT_EVENT = f"{SLUG}:snapshot"


def _metadata(message: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        key: value
        for key, value in message.items()
        if key not in ("content", "content_parts", "reasoning_summary")
    }
    # `content` is the last text part whenever parts exist; send it only when
    # it cannot be derived.
    parts = message.get("content_parts")
    texts = [p.get("text") or "" for p in parts or [] if p.get("type") == "text"]
    if parts is None or (texts[-1] if texts else "") != message.get("content"):
        metadata["content"] = message.get("content")
    if parts is None:
        metadata["content_parts"] = None
    return metadata


_TERMINAL_STATUS = {"dead": "incomplete", "errored": "failed"}

_TOOL_STATUS = {
    "running": "in_progress",
    "completed": "completed",
    "error": "incomplete",
}


def _item(message_id: str, index: int, part: Dict[str, Any], status: str):
    item_id = f"{message_id}_{index}"
    if part.get("type") == "text":
        return {
            "type": "message",
            "id": f"msg_{item_id}",
            "role": "assistant",
            "status": status,
            "content": [
                {
                    "type": "output_text",
                    "text": part.get("text") or "",
                    "annotations": [],
                }
            ],
        }
    if part.get("type") == "tool" and part.get("tool"):
        tool = part["tool"]
        return {
            "type": "function_call",
            "id": f"fc_{item_id}",
            "call_id": tool.get("tool_call_id") or item_id,
            "name": tool.get("name") or "",
            "arguments": json.dumps(tool.get("params") or {}),
            "status": _TOOL_STATUS.get(tool.get("status"), status),
            TOOL_FIELD: tool,
        }
    return {
        "type": f"{SLUG}:{part.get('type')}",
        "id": f"{SLUG}_{item_id}",
        "status": status,
        PART_FIELD: part,
    }


def _reasoning_item(message_id: str, summary: str, status: str):
    return {
        "type": "reasoning",
        "id": f"rs_{message_id}",
        "status": status,
        "summary": [{"type": "summary_text", "text": summary}] if summary else [],
    }


class OpenResponsesEncoder:
    """Turns one connection's cumulative chat messages into Open Responses events."""

    def __init__(self) -> None:
        self.__previous: Optional[Dict[str, Any]] = None
        self.__sequence = 0

    def __event(self, event_type: str, **fields: Any) -> Event:
        payload = {"type": event_type, "sequence_number": self.__sequence, **fields}
        self.__sequence += 1
        return event_type, payload

    def __response(self, message: Dict[str, Any], status: str) -> Dict[str, Any]:
        message_id = message.get("message_id") or ""
        # Output items have no "failed" status; a failed turn leaves them
        # incomplete.
        item_status = "incomplete" if status == "failed" else status
        output = [
            _reasoning_item(
                message_id, message.get("reasoning_summary") or "", item_status
            )
        ]
        output += [
            _item(message_id, i, part, item_status)
            for i, part in enumerate(message.get("content_parts") or [])
        ]
        return {
            "id": message_id,
            "object": "response",
            "status": status,
            "output": output,
            MESSAGE_FIELD: _metadata(message),
        }

    def encode(self, message: Dict[str, Any]) -> List[Event]:
        previous, self.__previous = self.__previous, message
        if previous is None:
            return [
                self.__event(
                    "response.created",
                    response=self.__response(message, "in_progress"),
                )
            ]
        events: List[Event] = []
        message_id = message.get("message_id") or ""
        old_parts = previous.get("content_parts") or []
        parts = message.get("content_parts") or []
        if len(parts) < len(old_parts) or any(
            (old.get("type"), old.get("tool", {}) and old["tool"].get("tool_call_id"))
            != (
                new.get("type"),
                new.get("tool", {}) and new["tool"].get("tool_call_id"),
            )
            for old, new in zip(old_parts, parts)
        ):
            return [
                self.__event(
                    SNAPSHOT_EVENT, response=self.__response(message, "in_progress")
                )
            ]

        old_summary = previous.get("reasoning_summary") or ""
        summary = message.get("reasoning_summary") or ""
        if summary != old_summary:
            if summary.startswith(old_summary):
                events.append(
                    self.__event(
                        "response.reasoning_summary_text.delta",
                        item_id=f"rs_{message_id}",
                        output_index=REASONING_INDEX,
                        summary_index=0,
                        delta=summary[len(old_summary) :],
                    )
                )
            else:
                events.append(
                    self.__event(
                        UPDATED_ITEM_EVENT,
                        output_index=REASONING_INDEX,
                        item=_reasoning_item(message_id, summary, "in_progress"),
                    )
                )

        for i, part in enumerate(parts):
            output_index = i + 1
            item = _item(message_id, i, part, "in_progress")
            if i >= len(old_parts):
                text = part.get("text") or "" if part.get("type") == "text" else ""
                added = _item(message_id, i, {**part, "text": ""}, "in_progress")
                events.append(
                    self.__event(
                        "response.output_item.added",
                        output_index=output_index,
                        item=added if part.get("type") == "text" else item,
                    )
                )
                if text:
                    events.append(self.__text_delta(item, output_index, text))
                continue
            old = old_parts[i]
            if old == part:
                continue
            old_text, text = old.get("text") or "", part.get("text") or ""
            if (
                part.get("type") == "text"
                and text.startswith(old_text)
                and {k: v for k, v in old.items() if k != "text"}
                == {k: v for k, v in part.items() if k != "text"}
            ):
                events.append(
                    self.__text_delta(item, output_index, text[len(old_text) :])
                )
            else:
                events.append(
                    self.__event(
                        UPDATED_ITEM_EVENT, output_index=output_index, item=item
                    )
                )

        old_metadata, metadata = _metadata(previous), _metadata(message)
        if metadata != old_metadata:
            changed = {k: v for k, v in metadata.items() if old_metadata.get(k) != v}
            removed = [k for k in old_metadata if k not in metadata]
            events.append(
                self.__event(
                    UPDATED_MESSAGE_EVENT,
                    set=changed,
                    **({"unset": removed} if removed else {}),
                )
            )
        return events

    def __text_delta(self, item: Dict[str, Any], output_index: int, delta: str):
        return self.__event(
            "response.output_text.delta",
            item_id=item["id"],
            output_index=output_index,
            content_index=0,
            delta=delta,
        )

    def finish(self, terminal_reason: Optional[str] = None) -> List[Event]:
        """``terminal_reason`` is the replayed turn's verdict: a turn whose
        producer died ends incomplete, one that failed ends failed."""
        if self.__previous is None:
            return []
        status = _TERMINAL_STATUS.get(terminal_reason or "", "completed")
        response = self.__response(self.__previous, status)
        events: List[Event] = []
        for output_index, item in enumerate(response["output"]):
            if item["type"] == "message":
                events.append(
                    self.__event(
                        "response.output_text.done",
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=0,
                        text=item["content"][0]["text"],
                    )
                )
            events.append(
                self.__event(
                    "response.output_item.done", output_index=output_index, item=item
                )
            )
        events.append(self.__event(f"response.{status}", response=response))
        return events
