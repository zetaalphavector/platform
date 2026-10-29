import json
from typing import List

from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion

TOOL_RESPONSE_PREVIEW_CHARS = 600


def render_transcript(transcript: List[ChatCompletion], max_chars: int) -> str:
    """One line per message, tool call and tool result, for a helper agent that
    reads another agent's trajectory. Long tool results are previewed; the whole
    rendering keeps its tail when it exceeds `max_chars`."""
    lines: List[str] = []
    for completion in transcript:
        role = completion.sender.value
        if completion.content:
            lines.append(f"[{role}] {completion.content}")
        for request in completion.tool_call_requests or []:
            call = request.function_call_request
            params = json.dumps(call.params or {}, default=str)
            lines.append(f"[{role} -> tool] {call.name}({params})")
        for response in completion.tool_call_responses or []:
            text = response.response or ""
            if len(text) > TOOL_RESPONSE_PREVIEW_CHARS:
                text = text[:TOOL_RESPONSE_PREVIEW_CHARS] + " ..."
            lines.append(f"[tool result {response.id}] {text}")
    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = "...\n" + rendered[-max_chars:]
    return rendered
