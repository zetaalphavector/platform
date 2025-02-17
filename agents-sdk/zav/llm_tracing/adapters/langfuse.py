from typing import Dict, Optional, Tuple, Union

import httpx
from langfuse import Langfuse
from langfuse.client import (
    StatefulGenerationClient,
    StatefulSpanClient,
    StatefulTraceClient,
)

from zav.llm_tracing.trace import Span, TracingBackend
from zav.llm_tracing.tracing_backend_factory import TracingBackendFactory
from zav.llm_tracing.tracing_configuration import LangfuseConfiguration


class LangfuseClientCache:
    # each key is a tuple of a hash of the config dict plus the public key, secret key,
    # and host
    __client_cache: Dict[Tuple[int, str, str, str], Langfuse] = {}

    @staticmethod
    def get_client(
        vendor_configuration: LangfuseConfiguration,
        httpx_client: Optional[httpx.Client],
    ) -> Langfuse:
        """Configure the Langfuse client.

        Args:
            vendor_configuration: Tracing configuration for Langfuse.
            httpx_client: Pass your own httpx client for more customizability
                of requests.
        """
        config_dict = tuple(sorted(vendor_configuration.dict())) + (id(httpx_client),)
        config_hash = hash(config_dict)
        cache_key = (
            config_hash,
            vendor_configuration.public_key,
            vendor_configuration.secret_key,
            vendor_configuration.host,
        )

        if cache_key not in LangfuseClientCache.__client_cache:
            LangfuseClientCache.__client_cache[cache_key] = Langfuse(
                public_key=vendor_configuration.public_key,
                secret_key=vendor_configuration.secret_key.get_unencrypted_secret(),
                host=vendor_configuration.host,
                release=vendor_configuration.release,
                debug=vendor_configuration.debug,
                threads=vendor_configuration.threads,
                flush_at=vendor_configuration.flush_at,
                flush_interval=vendor_configuration.flush_interval,
                max_retries=vendor_configuration.max_retries,
                timeout=vendor_configuration.timeout,
                sdk_integration=vendor_configuration.sdk_integration,
                httpx_client=httpx_client,
                enabled=vendor_configuration.enabled,
                sample_rate=vendor_configuration.sample_rate,
            )
        return LangfuseClientCache.__client_cache[cache_key]


@TracingBackendFactory.register("langfuse")
class LangfuseTracingBackend(TracingBackend):
    def __init__(
        self,
        vendor_configuration: LangfuseConfiguration,
        httpx_client: Optional[httpx.Client] = None,
    ):
        self.langfuse = LangfuseClientCache.get_client(
            vendor_configuration, httpx_client
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
