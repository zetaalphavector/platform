import zav.pydantic_compat._pydantic_v1_py314_fix  # isort: skip  # noqa: F401
import warnings
from typing import Dict, List, Optional, Tuple, Union

import httpx

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="'return' in a 'finally' block",
        category=SyntaxWarning,
    )
    from langfuse import Langfuse
    from langfuse.api.resources.commons.types import ScoreDataType
    from langfuse.api.resources.score.types import CreateScoreRequest
    from langfuse.client import (
        StatefulGenerationClient,
        StatefulSpanClient,
        StatefulTraceClient,
    )

from zav.logging import logger

from zav.llm_tracing.async_offload import submit_tracing_call
from zav.llm_tracing.feedback import FeedbackService
from zav.llm_tracing.feedback_service_factory import FeedbackServiceFactory
from zav.llm_tracing.trace import Span, TracingBackend
from zav.llm_tracing.trace_cleanup import TraceCleanupService
from zav.llm_tracing.trace_cleanup_service_factory import TraceCleanupServiceFactory
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
        submit_tracing_call(self.__handle_new_trace, span)

    def handle_new(self, span: Span):
        submit_tracing_call(self.__handle_new, span)

    def handle_update(self, span: Span):
        submit_tracing_call(self.__handle_update, span)

    def handle_event(self, span: Span):
        submit_tracing_call(self.__handle_event, span)

    def __handle_new_trace(self, span: Span):
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

    def __handle_new(self, span: Span):
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

    def __handle_update(self, span: Span):
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

    def __handle_event(self, span: Span):
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


@FeedbackServiceFactory.register("langfuse")
class LangfuseFeedbackService(FeedbackService):
    def __init__(
        self,
        vendor_configuration: LangfuseConfiguration,
        httpx_client: Optional[httpx.Client] = None,
    ):
        self.langfuse = LangfuseClientCache.get_client(
            vendor_configuration, httpx_client
        )

    async def add_score(
        self,
        trace_id: str,
        name: str,
        value: Union[str, float],
        comment: Optional[str] = None,
    ) -> None:
        """Add a score to a trace for user feedback.

        Args:
            trace_id: The ID of the trace to score (message_id).
            name: Name of the score (e.g., "user_feedback").
            value: The score value - string for CATEGORICAL, numeric for NUMERIC.
            comment: Optional comment/feedback text.
        """
        if isinstance(value, str):
            data_type = "CATEGORICAL"
        elif isinstance(value, float):
            data_type = "NUMERIC"
        else:
            raise ValueError(f"Invalid value type: {type(value)}")

        # Use trace_id as score_id to ensure one score per message
        score_id = trace_id

        request = CreateScoreRequest(
            traceId=trace_id,
            id=score_id,
            name=name,
            value=value,
            comment=comment,
            dataType=ScoreDataType(data_type),
        )

        await self.langfuse.async_api.score.create(request=request)

    async def validate_trace_ownership(
        self,
        trace_id: str,
        user_id: str,
    ) -> bool:
        """Validate that a trace belongs to the specified user.

        Args:
            trace_id: The ID of the trace to validate.
            user_id: The user ID to check ownership against.
        """
        try:
            trace = await self.langfuse.async_api.trace.get(trace_id)
            if not trace:
                return False

            return trace.user_id == user_id
        except Exception:
            return False


_DELETE_BATCH_SIZE = 50

# The deployed self-hosted Langfuse is v2, whose public (API-key) surface has no
# trace-delete endpoint: DELETE /api/public/traces returns 405, and deletion only
# exists via the UI's session-authenticated tRPC API. So delete_session_traces
# short-circuits below. Flip to True once on Langfuse v3, where the project API
# key can DELETE /api/public/traces and the implementation below works as-is.
_TRACE_DELETION_SUPPORTED = False


@TraceCleanupServiceFactory.register("langfuse")
class LangfuseTraceCleanupService(TraceCleanupService):
    def __init__(
        self,
        vendor_configuration: LangfuseConfiguration,
        httpx_client: Optional[httpx.Client] = None,
    ):
        self.langfuse = LangfuseClientCache.get_client(
            vendor_configuration, httpx_client
        )

    async def delete_session_traces(self, session_id: str) -> None:
        if not _TRACE_DELETION_SUPPORTED:
            # Skip the list+delete round trip that always 405s on Langfuse v2.
            # TODO: remove this guard (set _TRACE_DELETION_SUPPORTED = True) once
            # the server is upgraded to Langfuse v3.
            logger.warning(
                "Langfuse v2 has no trace-deletion API; skipping cleanup for "
                "session %s (requires Langfuse v3)",
                session_id,
            )
            return

        trace_ids: List[str] = []
        page = 1
        while True:
            traces = await self.langfuse.async_api.trace.list(
                session_id=session_id, page=page, limit=100
            )
            trace_ids.extend(item.id for item in traces.data)
            if not traces.data or page >= traces.meta.total_pages:
                break
            page += 1
        for i in range(0, len(trace_ids), _DELETE_BATCH_SIZE):
            await self.langfuse.async_api.trace.delete_multiple(
                trace_ids=trace_ids[i : i + _DELETE_BATCH_SIZE]
            )
