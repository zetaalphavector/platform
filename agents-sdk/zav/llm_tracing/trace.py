import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field  # type: ignore


def now():
    return datetime.now(timezone.utc)


def new_id():
    return str(uuid.uuid4())


class SpanContext(BaseModel):
    trace_id: str = Field(default_factory=new_id)
    span_id: str = Field(default_factory=new_id)
    trace_state: Dict[str, Any] = Field(default_factory=dict)


class SpanEvent(BaseModel):
    name: str
    timestamp: datetime = Field(default_factory=now)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class TracingBackend(ABC):
    @abstractmethod
    def __init__(self, vendor_configuration):
        raise NotImplementedError

    @abstractmethod
    def handle_new_trace(self, span: "Span"):
        raise NotImplementedError

    @abstractmethod
    def handle_new(self, span: "Span"):
        raise NotImplementedError

    @abstractmethod
    def handle_update(self, span: "Span"):
        raise NotImplementedError

    @abstractmethod
    def handle_event(self, span: "Span"):
        raise NotImplementedError


class Span(BaseModel):
    name: str
    context: SpanContext
    parent_id: Optional[str] = None
    start_time: datetime = Field(default_factory=now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[SpanEvent] = Field(default_factory=list)
    tracing_backend: TracingBackend = Field(..., exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def new(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        new_span = Span(
            name=name,
            context=SpanContext(
                trace_id=self.context.trace_id, trace_state=self.context.trace_state
            ),
            attributes=attributes or {},
            parent_id=self.context.span_id,
            events=[],
            tracing_backend=self.tracing_backend,
        )
        self.tracing_backend.handle_new(new_span)
        return new_span

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))
        self.tracing_backend.handle_event(self)
        return self

    def update(self, attributes: Optional[Dict[str, Any]] = None):
        self.attributes.update(attributes or {})
        self.tracing_backend.handle_update(self)
        return self

    def end(self, attributes: Optional[Dict[str, Any]] = None):
        self.end_time = now()
        return self.update(attributes)


class Trace:
    def __init__(self, tracing_backend: TracingBackend):
        self.tracing_backend = tracing_backend

    def new(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        trace_state: Optional[Dict[str, Any]] = None,
    ):
        span = Span(
            name=name,
            context=SpanContext(trace_state=trace_state or {}),
            attributes=attributes or {},
            events=[],
            tracing_backend=self.tracing_backend,
        )
        self.tracing_backend.handle_new_trace(span)
        return span
