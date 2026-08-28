import inspect
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple, Type

from zav.llm_domain import LLMClientConfiguration
from zav.llm_tracing import Span, Trace, TracingBackendFactory
from zav.pydantic_compat import _BaseModel

from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyRegistryProtocol,
    ResumableAgentDependency,
)
from zav.agents_sdk.domain.agent_event import AgentEvent
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetup, AgentSetupRetriever
from zav.agents_sdk.domain.chat_agent import ChatAgent, StreamableChatAgent
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistryProtocol
from zav.agents_sdk.domain.chat_request import ConversationContext
from zav.agents_sdk.domain.dependency_resolver import DependencyResolver
from zav.agents_sdk.domain.llm_configuration_store import LLMConfigurationStore


def init_span(
    tracing_backend_factory: Type[TracingBackendFactory],
    agent_setup: AgentSetup,
    agent_identifier: str,
    trace_state: Dict[str, Any],
    message_id: Optional[str] = None,
) -> Optional[Span]:
    span: Optional[Span] = None
    if tracing_config := agent_setup.tracing_configuration:
        tracing_backend = tracing_backend_factory.create(config=tracing_config)
        span = Trace(tracing_backend=tracing_backend).new(
            name="agent-response",
            attributes={
                "metadata": {"agent_identifier": agent_identifier},
            },
            trace_state=trace_state,
            trace_id=message_id,
        )

    return span


def init_sub_agent_span(
    agent_identifier: str, span: Optional[Span] = None
) -> Optional[Span]:
    if not span:
        return None

    return span.new(
        name="sub-agent-response",
        attributes={
            "metadata": {"agent_identifier": agent_identifier},
        },
    )


class ChatAgentFactory:
    """Creates configured agents. Dependency resolution itself lives in
    DependencyResolver; the factory supplies the two agent-shaped concerns the
    resolver delegates back: agent creation (the AgentCreation seam) and
    resumable-state tracking (the resolution hook)."""

    def __init__(
        self,
        agent_setup_retriever: AgentSetupRetriever,
        chat_agent_class_registry: ChatAgentClassRegistryProtocol,
        tracing_backend_factory: Type[TracingBackendFactory],
        trace_state_params: Dict[str, Any],
        agent_dependency_registry: Optional[AgentDependencyRegistryProtocol] = None,
        llm_configuration_store: Optional[LLMConfigurationStore] = None,
        debug_backend: Optional[Callable[[Any], Any]] = None,
        publish_event: Optional[
            Callable[[AgentEvent], Coroutine[None, None, None]]
        ] = None,
        message_id: Optional[str] = None,
        session_id: Optional[str] = None,
        dependency_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.__agent_setup_retriever = agent_setup_retriever
        self.__chat_agent_class_registry = chat_agent_class_registry
        self.__tracing_backend_factory = tracing_backend_factory
        self.__trace_state_params = trace_state_params
        self.__debug_backend = debug_backend
        self.__publish_event = publish_event
        self.__message_id = message_id
        self.__session_id = session_id
        self.__dependency_state = dependency_state
        self.__resumable_dependencies: Dict[str, ResumableAgentDependency] = {}
        self.__resolver = DependencyResolver(
            dependency_registry=agent_dependency_registry,
            agent_creation=self,
            on_dependency_resolved=self.__track_resumable_dependency,
            llm_configuration_store=llm_configuration_store,
        )
        self.agent_setup: Optional[AgentSetup] = None

    def __dependency_state_key(
        self,
        dependency_path: Tuple[str, ...],
        dependency: ResumableAgentDependency,
        is_singleton: bool,
    ) -> str:
        # A singleton is shared across the whole dependency tree and resolved once
        # per request, so its persisted state is keyed by ``state_key`` alone. This
        # keeps the key stable across turns even when the resolution order or the set
        # of paths that reference the singleton changes. A non-singleton is reached
        # through a single path, so the path keeps distinct instances apart.
        if is_singleton:
            return dependency.state_key
        return ".".join((*dependency_path, dependency.state_key))

    async def __track_resumable_dependency(
        self,
        dependency_path: Tuple[str, ...],
        dependency: Any,
        is_singleton: bool = False,
    ) -> None:
        if not isinstance(dependency, ResumableAgentDependency):
            return

        state_key = self.__dependency_state_key(
            dependency_path, dependency, is_singleton
        )
        already_tracked = self.__resumable_dependencies.get(state_key)
        if already_tracked is dependency:
            return
        if already_tracked is not None:
            raise ValueError(
                f"Resumable dependency state key collision for {state_key!r}: "
                f"{type(already_tracked).__name__} and {type(dependency).__name__} "
                "resolve to the same key."
            )
        if self.__dependency_state is not None and state_key in self.__dependency_state:
            await dependency.load(self.__dependency_state[state_key])
        self.__resumable_dependencies[state_key] = dependency

    async def dump_dependency_state(self) -> Dict[str, Dict[str, Any]]:
        state = dict(self.__dependency_state or {})
        for state_key, dependency in self.__resumable_dependencies.items():
            state[state_key] = await dependency.dump()
        return state

    @property
    def resumes_conversation(self) -> bool:
        # True only when the agent (or a sub-agent) carries a dependency that
        # replays the conversation transcript across turns — the resumable
        # chat-completion client. Other resumable state (e.g. a counter) restores
        # itself but NOT the conversation, so it does not make history work: such
        # an agent must still be re-fed the full stored transcript on a stateful
        # follow-up (see resolve_turn_conversation).
        return any(
            dependency.restores_conversation
            for dependency in self.__resumable_dependencies.values()
        )

    async def create_agent(
        self,
        agent_identifier: str,
        handler_params: Dict[str, Any],
        conversation_context: Optional[ConversationContext] = None,
        parent_span: Optional[Span] = None,
        extra_agent_kwargs: Optional[Dict[str, Any]] = None,
        track_resumable: bool = True,
    ) -> ChatAgent:
        """The resolver's AgentCreation seam: a dependency that demands an
        agent (AgentCreator, sub-agent injection) gets one under a sub-agent
        span of the requesting resolution."""
        return await self.create(
            agent_identifier=agent_identifier,
            handler_params=handler_params,
            conversation_context=conversation_context,
            span=init_sub_agent_span(
                span=parent_span, agent_identifier=agent_identifier
            ),
            extra_agent_kwargs=extra_agent_kwargs,
            track_resumable=track_resumable,
        )

    async def create(
        self,
        agent_identifier: str,
        handler_params: Dict[str, Any],
        conversation_context: Optional[ConversationContext] = None,
        span: Optional[Span] = None,
        extra_agent_kwargs: Optional[Dict[str, Any]] = None,
        track_resumable: bool = True,
    ) -> ChatAgent:
        agent_setup = await self.__agent_setup_retriever.get(
            agent_identifier=agent_identifier
        )
        if not agent_setup:
            raise ValueError(f"Unknown agent: {agent_identifier}")
        if self.agent_setup is None:
            self.agent_setup = agent_setup

        if span is None:
            span = init_span(
                tracing_backend_factory=self.__tracing_backend_factory,
                agent_setup=agent_setup,
                agent_identifier=agent_identifier,
                trace_state=self.__trace_state_params,
                message_id=self.__message_id,
            )
        agent_cls = await self.__chat_agent_class_registry.get(
            agent_name=agent_setup.agent_name
        )
        agent_cls_params = inspect.signature(agent_cls).parameters
        # An undeclared allowed list restricts runtime requests to the default.
        agent_cls_param_values = await self.__resolver.resolve_parameters(
            parameters=agent_cls_params,
            configuration=agent_setup.agent_configuration or {},
            handler_params=handler_params,
            key_prefix=agent_identifier,
            conversation_context=conversation_context,
            span=span,
            sub_agent_mapping=agent_setup.sub_agent_mapping,
            default_llm_configuration_name=agent_setup.llm_configuration_name,
            allowed_llm_configuration_names=(
                agent_setup.allowed_llm_configuration_names or ()
            ),
            track_resumable=track_resumable,
        )
        agent_instance = agent_cls(
            **{**agent_cls_param_values, **(extra_agent_kwargs or {})}
        )
        if agent_setup:
            agent_instance.agent_identifier = agent_setup.agent_identifier
        if span:
            span_agent_params = {
                param_name: param_value
                for param_name, param_value in agent_cls_param_values.items()
                if isinstance(
                    param_value,
                    (
                        int,
                        float,
                        str,
                        bool,
                        list,
                        tuple,
                        set,
                        dict,
                        type(None),
                        complex,
                    ),
                )
                or (
                    isinstance(param_value, _BaseModel)
                    and not isinstance(param_value, LLMClientConfiguration)
                )
            }
            span.update(
                attributes={
                    "metadata": {
                        **span.attributes.get("metadata", {}),
                        **span_agent_params,
                    }
                }
            )

        agent_instance.debug_backend = self.__debug_backend
        agent_instance.span = span
        agent_instance.message_id = self.__message_id
        agent_instance.session_id = self.__session_id

        if self.__publish_event:
            agent_instance.publish_event = self.__publish_event
        return agent_instance

    async def create_streamable(
        self,
        agent_identifier: str,
        handler_params: Dict[str, Any],
        conversation_context: Optional[ConversationContext] = None,
        span: Optional[Span] = None,
        extra_agent_kwargs: Optional[Dict[str, Any]] = None,
    ) -> StreamableChatAgent:
        agent_instance = await self.create(
            agent_identifier=agent_identifier,
            handler_params=handler_params,
            conversation_context=conversation_context,
            span=span,
            extra_agent_kwargs=extra_agent_kwargs,
        )
        if not isinstance(agent_instance, StreamableChatAgent):
            raise ValueError(f"Agent {agent_identifier} is not streamable")

        return agent_instance
