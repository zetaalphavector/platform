from typing import Any, Dict, List, Optional

from zav.llm_tracing import Span


def create_span(
    messages: List[Dict[str, Any]],
    model_name: str,
    model_temperature: float,
    max_tokens: int,
    stream: bool,
    json_output: Optional[bool] = None,
    tools_dict: Optional[Dict[str, Any]] = None,
    functions_dict: Optional[Dict[str, Any]] = None,
    interleave_system_message: Optional[str] = None,
    span: Optional[Span] = None,
) -> Optional[Span]:
    return (
        span.new(
            name="chat-completion",
            attributes={
                "observation_type": "generation",
                "model": model_name,
                "input": {
                    "messages": messages,
                    **(functions_dict or {}),
                    **(tools_dict or {}),
                },
                "model_parameters": {
                    "temperature": model_temperature,
                    "max_tokens": max_tokens,
                    **({"json_output": json_output} if json_output is not None else {}),
                    "interleave_system_message": interleave_system_message,
                    "stream": stream,
                },
            },
        )
        if span
        else None
    )


def end_span(
    usage: Dict,
    tool_calls: Optional[List[Any]] = None,
    span: Optional[Span] = None,
    content: Optional[str] = None,
    role: Optional[str] = None,
    function_call: Optional[Any] = None,
):
    if span:
        span.end(
            attributes={
                "output": {
                    "role": role,
                    "content": content,
                    **(
                        {"function_call": (function_call.dict())}
                        if function_call
                        else {}
                    ),
                    **(
                        {"tool_calls": [tool_call.dict() for tool_call in tool_calls]}
                        if tool_calls
                        else {}
                    ),
                },
                **usage,
            }
        )
