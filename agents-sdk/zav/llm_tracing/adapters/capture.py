from __future__ import annotations

from zav.llm_tracing.trace import Span, TracingBackend
from zav.llm_tracing.tracing_backend_factory import TracingBackendFactory
from zav.llm_tracing.tracing_configuration import CaptureConfiguration


@TracingBackendFactory.register("capture")
class CaptureTracingBackend(TracingBackend):
    """Tracing backend that records all spans and events in memory
    through a local store."""

    def __init__(self, vendor_configuration: CaptureConfiguration):
        self.__store = vendor_configuration.store

    def clear(self) -> None:
        self.__store.reset()

    def handle_new_trace(self, span: Span):
        self.__store.add_root_span(span)

    def handle_new(self, span: Span):
        self.__store.add_span(span)

    def handle_update(self, span: Span):
        self.__store.update_span(span)

    def handle_event(self, span: Span):
        self.__store.append_event(span)
