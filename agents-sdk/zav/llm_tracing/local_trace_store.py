from __future__ import annotations

from threading import RLock
from typing import Any

from zav.llm_tracing.trace import Span


class LocalTraceStore:
    """In-memory trace/span store. Thread-safe within a single process."""

    def __init__(self) -> None:
        self.__traces: dict[str, dict[str, Any]] = {}
        self.__spans: dict[str, dict[str, Any]] = {}
        self.__trace_to_spans: dict[str, list[str]] = {}
        self.__lock = RLock()

    def reset(self) -> None:
        with self.__lock:
            self.__traces.clear()
            self.__spans.clear()
            self.__trace_to_spans.clear()

    def add_root_span(self, span: Span) -> None:
        data = span.dict()
        span_id = span.context.span_id
        trace_id = span.context.trace_id
        with self.__lock:
            self.__traces[span_id] = data
            self.__spans[span_id] = data
            self.__trace_to_spans.setdefault(trace_id, [])

    def add_span(self, span: Span) -> None:
        data = span.dict()
        span_id = span.context.span_id
        trace_id = span.context.trace_id
        with self.__lock:
            self.__spans[span_id] = data
            self.__trace_to_spans.setdefault(trace_id, []).append(span_id)

    def update_span(self, span: Span) -> None:
        with self.__lock:
            rec = self.__spans.get(span.context.span_id)
            if not rec:
                return
            rec["attributes"] = span.attributes
            rec["end_time"] = span.end_time

    def append_event(self, span: Span) -> None:
        with self.__lock:
            rec = self.__spans.get(span.context.span_id)
            if not rec:
                return
            if "events" not in rec or rec["events"] is None:
                rec["events"] = []
            if span.events:
                rec["events"].append(span.events[-1].dict())

    def get_traces(self) -> dict[str, dict[str, Any]]:
        with self.__lock:
            return {k: v for k, v in self.__traces.items()}

    def get_spans(self) -> dict[str, dict[str, Any]]:
        with self.__lock:
            return {k: v for k, v in self.__spans.items()}

    def get_trace_spans(self, trace_id: str) -> list[dict[str, Any]]:
        with self.__lock:
            span_ids = self.__trace_to_spans.get(trace_id, [])
            return [self.__spans[sid] for sid in span_ids if sid in self.__spans]

    def get_trace(self, span_id: str) -> dict[str, Any] | None:
        with self.__lock:
            return self.__traces.get(span_id)

    def get_span(self, span_id: str) -> dict[str, Any] | None:
        with self.__lock:
            return self.__spans.get(span_id)

    def export(self) -> dict[str, Any]:
        with self.__lock:
            return {
                "traces": {k: v for k, v in self.__traces.items()},
                "spans": {k: v for k, v in self.__spans.items()},
                "trace_to_spans": {
                    k: list(v) for k, v in self.__trace_to_spans.items()
                },
            }
