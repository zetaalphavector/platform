import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from zav.agents_sdk.domain.agent_creator import SubAgentToolResult
from zav.agents_sdk.domain.chat_message import ContentPartTool
from zav.agents_sdk.domain.tools import Tool, apply_transform, format_display_text


def parse_result_dict(exec_response: Any) -> Optional[dict]:
    if isinstance(exec_response, SubAgentToolResult):
        return {"text": exec_response.text}
    if isinstance(exec_response, dict):
        return exec_response
    if isinstance(exec_response, str):
        try:
            parsed = json.loads(exec_response)
        except (json.JSONDecodeError, ValueError):
            return None
        # A JSON array or scalar result is text to the model, not a response object.
        return parsed if isinstance(parsed, dict) else None
    return None


def llm_tool_result(tool: Optional[Tool], exec_response: Any) -> Any:
    if isinstance(exec_response, SubAgentToolResult):
        return exec_response.text
    llm_transform = (
        tool.streaming_config.llm_response_transform
        if tool is not None and tool.streaming_config is not None
        else None
    )
    if llm_transform is None:
        return exec_response
    return apply_transform(parse_result_dict(exec_response), llm_transform)


def _format_duration(seconds: float) -> str:
    if seconds < 0.1:
        return "<0.1 s"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {rest} s"
    return f"{minutes // 60} h {minutes % 60} min"


def tool_running_part(
    tool: Optional[Tool], tool_call_id: str, params: Optional[Dict[str, Any]]
) -> Optional[ContentPartTool]:
    if tool is None or tool.streaming_config is None:
        return None
    streaming_config = tool.streaming_config
    return ContentPartTool(
        tool_call_id=tool_call_id,
        name=tool.name,
        params=apply_transform(params, streaming_config.params_transform),
        display_text=format_display_text(
            streaming_config.running_text,
            params,
            defaults=tool.get_parameter_defaults(),
        ),
        status="running",
        display=tool.display,
    )


def tool_completion_part(
    tool: Optional[Tool],
    tool_call_id: str,
    params: Optional[Dict[str, Any]],
    exec_response: Any,
    failed: bool,
    duration_seconds: Optional[float] = None,
    replaced_by_rule: bool = False,
    rule_reason: Optional[str] = None,
) -> Optional[ContentPartTool]:
    if tool is None or tool.streaming_config is None:
        return None
    streaming_config = tool.streaming_config
    visible_params = apply_transform(params, streaming_config.params_transform)
    if failed:
        return ContentPartTool(
            tool_call_id=tool_call_id,
            name=tool.name,
            params=visible_params,
            display_text=None,
            status="error",
            display=tool.display,
        )
    result_dict = parse_result_dict(exec_response)
    return ContentPartTool(
        tool_call_id=tool_call_id,
        name=tool.name,
        params=visible_params,
        response=apply_transform(result_dict, streaming_config.response_transform),
        display_text=format_display_text(
            streaming_config.completed_text or streaming_config.running_text,
            params,
            result_dict,
            defaults={
                **tool.get_parameter_defaults(),
                **(
                    {"duration": _format_duration(duration_seconds)}
                    if duration_seconds is not None
                    else {}
                ),
            },
        ),
        status="completed",
        display=tool.display,
        replaced_by_rule=replaced_by_rule,
        rule_reason=rule_reason,
    )


async def stream_tool_call(
    emit: Callable[[ContentPartTool], None],
    tool_call_id: str,
    tool: Optional[Tool],
    params: Optional[Dict[str, Any]],
    call: Awaitable[Any],
) -> Any:
    def emit_part(part: Optional[ContentPartTool]) -> None:
        if part is not None:
            emit(part)

    emit_part(tool_running_part(tool, tool_call_id, params))
    started = time.monotonic()
    try:
        result = await call
    except (Exception, asyncio.CancelledError):
        emit_part(tool_completion_part(tool, tool_call_id, params, None, failed=True))
        raise
    emit_part(
        tool_completion_part(
            tool,
            tool_call_id,
            params,
            result,
            failed=False,
            duration_seconds=time.monotonic() - started,
        )
    )
    if isinstance(result, SubAgentToolResult):
        # A sub-agent streams into its own runner; its calls arrive when it returns.
        for part in result.content_parts:
            if part.type == "tool" and part.tool is not None:
                emit(
                    part.tool.model_copy(
                        update={
                            "tool_call_id": f"{tool_call_id}:{part.tool.tool_call_id}"
                        }
                    )
                )
    return result
