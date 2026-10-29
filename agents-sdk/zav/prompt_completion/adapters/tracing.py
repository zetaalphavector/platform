from typing import Any, Dict, List, Optional

from zav.llm_tracing import Span


def create_span(
    messages: List[Dict[str, Any]],
    model_name: str,
    model_temperature: Optional[float],
    max_tokens: Optional[int],
    stream: bool,
    api_endpoint: Optional[str] = None,
    json_output: Optional[bool] = None,
    tools_dict: Optional[Dict[str, Any]] = None,
    functions_dict: Optional[Dict[str, Any]] = None,
    interleave_system_message: Optional[str] = None,
    span: Optional[Span] = None,
    tool_search: Optional[str] = None,
) -> Optional[Span]:
    return (
        span.new(
            name="chat-completion",
            attributes={
                "observation_type": "generation",
                **({"api_endpoint": api_endpoint} if api_endpoint is not None else {}),
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
                    # The vendor tool search protocol the deferred tools used.
                    **({"tool_search": tool_search} if tool_search else {}),
                },
            },
        )
        if span
        else None
    )


def create_image_generation_span(
    prompt: str,
    model_name: str,
    model_parameters: Dict[str, Any],
    span: Optional[Span] = None,
) -> Optional[Span]:
    return (
        span.new(
            name="image-generation",
            attributes={
                "observation_type": "generation",
                "model": model_name,
                "input": {"prompt": prompt},
                "model_parameters": model_parameters,
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
    log_probs: Optional[List[Any]] = None,
    reasoning_summary: Optional[str] = None,
    model: Optional[str] = None,
):
    if span:
        span.end(
            attributes={
                # The model that served the request. Differs from the one the span
                # was opened with behind routing deployments (e.g. Azure model
                # router), and Langfuse prices the generation by this value.
                **({"model": model} if model else {}),
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
                    **({"log_probs": log_probs} if log_probs else {}),
                    **(
                        {"reasoning_summary": reasoning_summary}
                        if reasoning_summary
                        else {}
                    ),
                },
                **usage,
            }
        )
