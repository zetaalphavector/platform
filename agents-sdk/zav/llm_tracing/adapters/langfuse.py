from typing import Dict, Optional, Union

import httpx
from langfuse import Langfuse
from langfuse.client import (
    StatefulGenerationClient,
    StatefulSpanClient,
    StatefulTraceClient,
)

from zav.llm_tracing.trace import Span, TracingBackend
from zav.llm_tracing.tracing_backend_factory import TracingBackendFactory


@TracingBackendFactory.register("langfuse")
class LangfuseTracingBackend(TracingBackend):
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        release: Optional[str] = None,
        debug: bool = False,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[float] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,  # seconds
        sdk_integration: Optional[str] = "default",
        httpx_client: Optional[httpx.Client] = None,
        enabled: Optional[bool] = True,
        sample_rate: Optional[float] = None,
    ):
        """Configure the Langfuse client.

        Args:
            public_key: Public API key of Langfuse project.
            secret_key: Secret API key of Langfuse project.
            host: Host of Langfuse API. Defaults to `https://cloud.langfuse.com`.
            release: Release number/hash of the application to provide analytics
                grouped by release.
            debug: Enables debug mode for more verbose logging.
            threads: Number of consumer threads to execute network requests.
                Helps scaling the SDK for high load. Only increase this if you run
                into scaling issues.
            flush_at: Max batch size that's sent to the API.
            flush_interval: Max delay until a new batch is sent to the API.
            max_retries: Max number of retries in case of API/network errors.
            timeout: Timeout of API requests in seconds. Defaults to 20 seconds.
            httpx_client: Pass your own httpx client for more customizability
                of requests.
            sdk_integration: Used by intgerations that wrap the Langfuse SDK to
                add context for debugging and support. Not to be used directly.
            enabled: Enables or disables the Langfuse client.
            sample_rate: Sampling rate for tracing. If set to 0.2, only 20% of the
                data will be sent to the backend.
        """
        self.langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            release=release,
            debug=debug,
            threads=threads,
            flush_at=flush_at,
            flush_interval=flush_interval,
            max_retries=max_retries,
            timeout=timeout,
            sdk_integration=sdk_integration,
            httpx_client=httpx_client,
            enabled=enabled,
            sample_rate=sample_rate,
        )
        self.__observations_map: Dict[
            str,
            Union[StatefulSpanClient, StatefulGenerationClient, StatefulTraceClient],
        ] = {}

    def handle_new_trace(self, span: Span):
        observation = self.langfuse.trace(
            id=span.context.trace_id,
            name=span.name,
            user_id=span.context.trace_state.get("user_id"),
            session_id=span.context.trace_state.get("session_id"),
            version=span.context.trace_state.get("version"),
            release=span.context.trace_state.get("release"),
            input=span.attributes.get("input"),
            output=span.attributes.get("output"),
            metadata=span.attributes.get("metadata"),
            tags=span.attributes.get("tags"),
            timestamp=span.start_time,
            public=False,
            **{
                k: v
                for k, v in span.attributes.items()
                if k not in {"input", "output", "metadata", "tags"}
            },
        )
        self.__observations_map[span.context.span_id] = observation

    def handle_new(self, span: Span):
        observation_type = span.attributes.get("observation_type")
        if observation_type == "generation":
            observation = self.langfuse.generation(
                id=span.context.span_id,
                trace_id=span.context.trace_id,
                parent_observation_id=span.parent_id,
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                completion_start_time=span.attributes.get("completion_start_time"),
                metadata=span.attributes.get("metadata"),
                level=span.attributes.get("level"),
                status_message=span.attributes.get("status_message"),
                version=span.attributes.get("version"),
                model=span.attributes.get("model"),
                model_parameters=span.attributes.get("model_parameters"),
                input=span.attributes.get("input"),
                output=span.attributes.get("output"),
                usage=span.attributes.get("usage"),
                prompt=span.attributes.get("prompt"),
                **{
                    k: v
                    for k, v in span.attributes.items()
                    if k
                    not in {
                        "completion_start_time",
                        "metadata",
                        "level",
                        "status_message",
                        "version",
                        "model",
                        "model_parameters",
                        "input",
                        "output",
                        "usage",
                        "prompt",
                    }
                },
            )
        else:
            observation = self.langfuse.span(
                id=span.context.span_id,
                trace_id=span.context.trace_id,
                parent_observation_id=span.parent_id,
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                metadata=span.attributes.get("metadata"),
                level=span.attributes.get("level"),
                status_message=span.attributes.get("status_message"),
                input=span.attributes.get("input"),
                output=span.attributes.get("output"),
                version=span.attributes.get("version"),
                **{
                    k: v
                    for k, v in span.attributes.items()
                    if k
                    not in {
                        "metadata",
                        "level",
                        "status_message",
                        "input",
                        "output",
                        "version",
                    }
                },
            )
        self.__observations_map[span.context.span_id] = observation

    def handle_update(self, span: Span):
        observation = self.__observations_map.get(span.context.span_id)
        if not observation:
            return

        if isinstance(observation, StatefulSpanClient):
            observation.update(
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                metadata=span.attributes.get("metadata"),
                level=span.attributes.get("level"),
                status_message=span.attributes.get("status_message"),
                input=span.attributes.get("input"),
                output=span.attributes.get("output"),
                version=span.attributes.get("version"),
                **{
                    k: v
                    for k, v in span.attributes.items()
                    if k
                    not in {
                        "metadata",
                        "level",
                        "status_message",
                        "input",
                        "output",
                        "version",
                    }
                },
            )
        elif isinstance(observation, StatefulGenerationClient):
            observation.update(
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                completion_start_time=span.attributes.get("completion_start_time"),
                metadata=span.attributes.get("metadata"),
                level=span.attributes.get("level"),
                status_message=span.attributes.get("status_message"),
                version=span.attributes.get("version"),
                model=span.attributes.get("model"),
                model_parameters=span.attributes.get("model_parameters"),
                input=span.attributes.get("input"),
                output=span.attributes.get("output"),
                usage=span.attributes.get("usage"),
                prompt=span.attributes.get("prompt"),
                **{
                    k: v
                    for k, v in span.attributes.items()
                    if k
                    not in {
                        "completion_start_time",
                        "metadata",
                        "level",
                        "status_message",
                        "version",
                        "model",
                        "model_parameters",
                        "input",
                        "output",
                        "usage",
                        "prompt",
                    }
                },
            )
        elif isinstance(observation, StatefulTraceClient):
            observation.update(
                name=span.name,
                user_id=span.context.trace_state.get("user_id"),
                session_id=span.context.trace_state.get("session_id"),
                version=span.context.trace_state.get("version"),
                release=span.context.trace_state.get("release"),
                input=span.attributes.get("input"),
                output=span.attributes.get("output"),
                metadata=span.attributes.get("metadata"),
                tags=span.attributes.get("tags"),
                public=False,
                **{
                    k: v
                    for k, v in span.attributes.items()
                    if k not in {"input", "output", "metadata", "tags"}
                },
            )

    def handle_event(self, span: Span):
        last_event = span.events[-1]
        self.langfuse.event(
            trace_id=span.context.trace_id,
            parent_observation_id=span.parent_id,
            name=last_event.name,
            start_time=last_event.timestamp,
            metadata=last_event.attributes.get("metadata"),
            input=last_event.attributes.get("input"),
            output=last_event.attributes.get("output"),
            level=last_event.attributes.get("level"),
            status_message=last_event.attributes.get("status_message"),
            version=last_event.attributes.get("version"),
            **{
                k: v
                for k, v in last_event.attributes.items()
                if k
                not in {
                    "metadata",
                    "input",
                    "output",
                    "level",
                    "status_message",
                    "version",
                }
            },
        )
