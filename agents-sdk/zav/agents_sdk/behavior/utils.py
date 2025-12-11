from typing import Any

from zav.llm_tracing import LocalTraceStore

from zav.agents_sdk.domain.chat_message import FunctionCallRequest


def parse_tool_calls(trace_store: LocalTraceStore) -> list[FunctionCallRequest]:
    calls: list[FunctionCallRequest] = []
    traces = trace_store.get_traces()
    if traces:
        latest_trace_span = max(traces.values(), key=lambda t: t.get("start_time") or 0)
        trace_id = latest_trace_span["context"]["trace_id"]
        span_dicts = trace_store.get_trace_spans(trace_id)
        for sp in span_dicts:
            attrs: dict[str, Any] = sp.get("attributes", {}) or {}
            metadata = attrs.get("metadata") or {}
            if isinstance(metadata, dict) and "tool_call_id" in metadata:
                calls.append(
                    FunctionCallRequest(
                        name=str(sp.get("name")),
                        params=attrs.get("input"),
                    )
                )

    return calls
