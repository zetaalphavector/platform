from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from zav.llm_tracing import LocalTraceStore
from zav.logging import logger

from zav.agents_sdk.behavior.models import (
    CitationExpectation,
    Expectation,
    JsonIncludesExpectation,
    TextIncludesExpectation,
    ToolCallExpectation,
)
from zav.agents_sdk.domain.chat_message import (
    ChatMessage,
    ChatMessageEvidence,
    FunctionCallRequest,
)


class AssertionRegistry:
    _registry: Dict[
        str, Callable[[LocalTraceStore, ChatMessage, Expectation], None]
    ] = {}

    @classmethod
    def register(cls, expectation_type: str):
        def decorator(func):
            cls._registry[expectation_type] = func
            return func

        return decorator

    @classmethod
    def evaluate_response(
        cls,
        trace_store: LocalTraceStore,
        response: ChatMessage,
        expectations: List[Expectation],
    ):
        for exp in expectations:
            handler = cls._registry.get(exp.type)
            if handler is None:
                raise NotImplementedError(f"No assertion handler for type '{exp.type}'")
            handler(trace_store, response, exp)


@AssertionRegistry.register("text_includes")
def _assert_text_includes(
    trace_store: LocalTraceStore,
    response: ChatMessage,
    expectation: TextIncludesExpectation,
):
    if expectation.value not in response.content:
        raise AssertionError(
            f"Text '{expectation.value}' not in response '{response.content}'."
        )


@AssertionRegistry.register("json_includes")
def _assert_json_includes(
    trace_store: LocalTraceStore,
    response: ChatMessage,
    expectation: JsonIncludesExpectation,
):
    expected = expectation.items
    actual = json.loads(response.content)
    missing_keys = [key for key in expected.keys() if key not in actual.keys()]
    if missing_keys:
        raise AssertionError(
            f"Expected keys '{missing_keys}' not in response '{actual}'."
        )
    for key, value in expected.items():
        if actual[key] != value:
            raise AssertionError(
                f"Value for key '{key}' in response '{response.content}' does not "
                f"match expected value '{value}'"
            )


def _p_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        # partial match for strings
        return expected in actual
    elif isinstance(expected, list) and isinstance(actual, list):
        # partial match for lists
        for expected_item in expected:
            if not any(_p_match(expected_item, actual_item) for actual_item in actual):
                return False
        return True
    elif not isinstance(expected, dict) or not isinstance(actual, dict):
        # exact match for other types (eg numbers, booleans)
        return expected == actual
    # partial match for dicts
    for key, val in expected.items():
        if key not in actual or not _p_match(val, actual[key]):
            return False
    return True


@AssertionRegistry.register("tool_call")
def _assert_tool_call(
    trace_store: LocalTraceStore,
    response: ChatMessage,
    expectation: ToolCallExpectation,
):
    calls: List[FunctionCallRequest] = []
    if response.function_call_request is not None:
        calls.append(response.function_call_request)

    # Derive tool calls from the in-memory capture tracing backend
    try:
        traces = trace_store.get_traces()
        if traces:
            latest_trace_span = max(
                traces.values(), key=lambda t: t.get("start_time") or 0
            )
            trace_id = latest_trace_span["context"]["trace_id"]
            span_dicts = trace_store.get_trace_spans(trace_id)
            for sp in span_dicts:
                attrs: Dict[str, Any] = sp.get("attributes", {}) or {}
                metadata = attrs.get("metadata") or {}
                if isinstance(metadata, dict) and "tool_call_id" in metadata:
                    calls.append(
                        FunctionCallRequest(
                            name=str(sp.get("name")),
                            params=attrs.get("input"),
                        )
                    )
    except Exception:
        logger.warning(
            "Failed to derive tool calls from tracing backend", exc_info=True
        )

    if not calls:
        raise AssertionError(
            "Expected at least one function/tool call but none was found."
        )

    def _matches(call: FunctionCallRequest) -> bool:
        if call.name != expectation.name:
            return False
        if expectation.params_partial:
            actual_params: Dict[str, Any] = call.params or {}
            for key, val in expectation.params_partial.items():
                if key not in actual_params or not _p_match(val, actual_params[key]):
                    return False
        return True

    if not any(_matches(call) for call in calls):
        raise AssertionError(
            f"No tool call matched expectation. Found: {[call.name for call in calls]}"
        )


@AssertionRegistry.register("citation")
def _assert_citations(
    trace_store: LocalTraceStore,
    response: ChatMessage,
    expectation: CitationExpectation,
):
    evidences: List[ChatMessageEvidence] = response.evidences or []
    count = len(evidences)
    if expectation.min_citations is not None and count < expectation.min_citations:
        raise AssertionError(
            f"Expected at least {expectation.min_citations} citations, got {count}."
        )
    if expectation.cited_docs is not None:
        cited_doc_ids = []
        for e in evidences:
            doc_id = e.document_hit_url.split("property_values=")[1]
            doc_id = doc_id.split("_")[0]  # convert to guid
            cited_doc_ids.append(doc_id)
        if not all(doc in cited_doc_ids for doc in expectation.cited_docs):
            raise AssertionError(
                "Expected the following documents to be cited: "
                f"{expectation.cited_docs}, but got: {cited_doc_ids}."
            )
