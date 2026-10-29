from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Dict

from zav.llm_tracing.local_trace_store import LocalTraceStore
from zav.llm_tracing.trace import Span, TracingBackend
from zav.llm_tracing.trace_cleanup import TraceCleanupService
from zav.llm_tracing.trace_cleanup_service_factory import TraceCleanupServiceFactory
from zav.llm_tracing.tracing_backend_factory import TracingBackendFactory
from zav.llm_tracing.tracing_configuration import FileConfiguration


@TracingBackendFactory.register("file")
class FileTracingBackend(TracingBackend):
    """Tracing backend that writes every trace to its own JSON file, in the
    `LocalTraceStore` export format. The file is rewritten whenever a span is
    created or updated, so a run that dies mid-way still leaves its last state on
    disk; events only buffer until the next update."""

    def __init__(self, vendor_configuration: FileConfiguration):
        self.__directory = Path(vendor_configuration.directory)
        self.__directory.mkdir(parents=True, exist_ok=True)
        self.__file_name = vendor_configuration.file_name
        self.__store = LocalTraceStore()
        self.__paths: Dict[str, Path] = {}
        self.__lock = RLock()

    def handle_new_trace(self, span: Span):
        self.__store.add_root_span(span)
        with self.__lock:
            self.__paths[span.context.trace_id] = self.__directory / self.__name(span)
        self.__write(span.context.trace_id)

    def __name(self, span: Span) -> str:
        trace_id = span.context.trace_id
        if self.__file_name == "trace_id":
            return f"{trace_id}.trace.json"
        return f"{span.start_time:%Y-%m-%dT%H-%M-%S}_{trace_id}.trace.json"

    def handle_new(self, span: Span):
        self.__store.add_span(span)
        self.__write(span.context.trace_id)

    def handle_update(self, span: Span):
        self.__store.update_span(span)
        self.__write(span.context.trace_id)

    def handle_event(self, span: Span):
        self.__store.append_event(span)

    def __write(self, trace_id: str) -> None:
        with self.__lock:
            path = self.__paths.get(trace_id)
            if path is None:
                return
            payload = json.dumps(self.__export(trace_id), indent=2, default=str)
            temp_path = path.with_name(f"{path.name}.tmp")
            temp_path.write_text(payload)
            os.replace(temp_path, path)

    def __export(self, trace_id: str) -> Dict[str, Any]:
        export = self.__store.export()
        spans = {
            span_id: span
            for span_id, span in export["spans"].items()
            if span["context"]["trace_id"] == trace_id
        }
        return {
            "traces": {
                span_id: trace
                for span_id, trace in export["traces"].items()
                if span_id in spans
            },
            "spans": spans,
            "trace_to_spans": {trace_id: export["trace_to_spans"].get(trace_id, [])},
        }


@TraceCleanupServiceFactory.register("file")
class FileTraceCleanupService(TraceCleanupService):
    def __init__(self, vendor_configuration: FileConfiguration):
        pass

    async def delete_session_traces(self, session_id: str) -> None:
        return
