import inspect
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Optional,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

from zav.llm_domain import LLMClientConfiguration
from zav.llm_tracing import Span, Trace, TracingBackendFactory
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.domain.agent_creator import AgentCreator
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistryProtocol
from zav.agents_sdk.domain.agent_event import AgentEvent
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetup, AgentSetupRetriever
from zav.agents_sdk.domain.chat_agent import ChatAgent, StreamableChatAgent
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistryProtocol
from zav.agents_sdk.domain.chat_request import ConversationContext
from zav.agents_sdk.security import sanitize_bot_params


def check_is_optional(field):
    origin = get_origin(field)
    return origin is Union and type(None) in get_args(field)


def check_is_class(annotation):
    return inspect.isclass(annotation)


def init_span(
    tracing_backend_factory: Type[TracingBackendFactory],
    agent_setup: AgentSetup,
    agent_identifier: str,
    trace_state: Dict[str, Any],
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
    def __init__(
        self,
        agent_setup_retriever: AgentSetupRetriever,
        chat_agent_class_registry: ChatAgentClassRegistryProtocol,
        tracing_backend_factory: Type[TracingBackendFactory],
        trace_state_params: Dict[str, Any],
        agent_dependency_registry: Optional[AgentDependencyRegistryProtocol] = None,
        debug_backend: Optional[Callable[[Any], Any]] = None,
        publish_event: Optional[
            Callable[[AgentEvent], Coroutine[None, None, None]]
        ] = None,
    ):
        self.__agent_setup_retriever = agent_setup_retriever
        self.__chat_agent_class_registry = chat_agent_class_registry
        self.__tracing_backend_factory = tracing_backend_factory
        self.__trace_state_params = trace_state_params
        self.__agent_dependency_registry = agent_dependency_registry
        self.__debug_backend = debug_backend
        self.__publish_event = publish_event
        self.agent_setup: Optional[AgentSetup] = None

    async def __parse_sub_agent(
        self,
        has_default: bool,
        is_optional: bool,
        sub_agent_name: str,
        sub_agent_identifier: str,
        param_default: Any,
        handler_params: Dict[str, Any],
        conversation_context: Optional[ConversationContext] = None,
        span: Optional[Span] = None,
    ):
        try:
            return await self.create(
                agent_identifier=sub_agent_identifier,
                handler_params=handler_params,
                agent_name=sub_agent_name,
                conversation_context=conversation_context,
                span=init_sub_agent_span(
                    span=span, agent_identifier=sub_agent_identifier
                ),
            )
        except ValueError as e:
            if has_default:
                return param_default
            elif is_optional:
                return None
            else:
                raise e

    def __parse_agent_configuration(
        self,
        param_name: str,
        has_default: bool,
        is_optional: bool,
        is_not_annotated: bool,
        is_base_model: bool,
        param_default: Any,
        param_annotation: BaseModel,
        handler_params: Dict[str, Any],
        agent_setup: Optional[AgentSetup] = None,
    ):
        agent_configuration = (
            agent_setup.agent_configuration
            if agent_setup and agent_setup.agent_configuration
            else {}
        )
        is_param_missing = param_name not in agent_configuration
        param_value = agent_configuration.get(param_name, None)

        if is_param_missing:
            if handler_params and param_name in handler_params:
                param_value = handler_params[param_name]
            elif has_default:
                return param_default
            else:
                raise ValueError(f"Missing value for required parameter: {param_name}")
        if param_value is None:
            if is_optional:
                # The arg is optional and the value is None so we return None
                return None
            else:
                # The arg is not optional and the value is None so we raise an error
                raise ValueError(f"Missing value for required parameter: {param_name}")
        if is_not_annotated:
            # The arg is not typed so we return the value as is
            return param_value
        elif is_base_model:
            # Try to parse the value as a Pydantic model
            if isinstance(param_value, dict):
                return param_annotation.parse_obj(param_value)
            elif isinstance(param_value, str):
                return param_annotation.parse_raw(param_value)
            elif isinstance(param_value, BaseModel):
                return param_annotation.from_orm(param_value)
            else:
                raise ValueError(
                    f"Unsupported type for {param_name}: {type(param_value)}"
                )
        else:
            # We assume this is a value that can be directly passed
            return param_value

    async def __parse_value(
        self,
        param: inspect.Parameter,
        param_name: str,
        handler_params: Dict[str, Any],
        conversation_context: Optional[ConversationContext] = None,
        agent_setup: Optional[AgentSetup] = None,
        span: Optional[Span] = None,
    ) -> Optional[Any]:
        param_annotation = param.annotation
        is_optional = check_is_optional(param_annotation)
        if is_optional:
            param_annotation = next(
                annotation
                for annotation in get_args(param_annotation)
                if annotation is not type(None)  # noqa: E721
            )
        is_class = inspect.isclass(param_annotation)
        # Parse AgentCreator
        if is_class and issubclass(param_annotation, AgentCreator):

            async def agent_factory(
                factory_agent_identifier: str,
                bot_params: Dict[str, Any],
                factory_conversation_context: Optional[ConversationContext] = None,
                extra_agent_kwargs: Optional[Dict[str, Any]] = None,
            ):
                return await self.create(
                    agent_identifier=factory_agent_identifier,
                    handler_params={
                        **handler_params,
                        **sanitize_bot_params(bot_params),
                    },
                    conversation_context=factory_conversation_context,
                    span=init_sub_agent_span(
                        span=span, agent_identifier=factory_agent_identifier
                    ),
                    extra_agent_kwargs=extra_agent_kwargs,
                )

            return AgentCreator(agent_factory=agent_factory)
        # Parse agent dependency
        if self.__agent_dependency_registry and is_class:
            agent_dependency = self.__agent_dependency_registry.get(param_annotation)
            if agent_dependency:
                # The agent dependency needs to be inspected and initialized
                agent_dependency_params = inspect.signature(
                    agent_dependency.create
                ).parameters
                return agent_dependency.create(
                    **{
                        param_name: (
                            await self.__parse_value(
                                param=param,
                                param_name=param_name,
                                handler_params=handler_params,
                                conversation_context=conversation_context,
                                agent_setup=agent_setup,
                                span=span,
                            )
                        )
                        for param_name, param in agent_dependency_params.items()
                        if param_name != "self"
                    }
                )
        has_default = param.default != inspect.Parameter.empty
        # parse conversation context
        is_conversation_context = is_class and issubclass(
            param_annotation, ConversationContext
        )
        if is_conversation_context:
            return conversation_context
        # Parse sub agent
        is_chat_agent = is_class and issubclass(param_annotation, ChatAgent)
        if is_chat_agent:
            # Retrieve agent_identifier from agent_setup
            sub_agent_name = cast(ChatAgent, param_annotation).agent_name
            sub_agent_identifier = sub_agent_name
            if agent_setup and agent_setup.sub_agent_mapping:
                sub_agent_identifier = agent_setup.sub_agent_mapping.get(
                    sub_agent_name, sub_agent_name
                )
            return await self.__parse_sub_agent(
                has_default=has_default,
                is_optional=is_optional,
                sub_agent_name=sub_agent_name,
                sub_agent_identifier=sub_agent_identifier,
                param_default=param.default,
                handler_params=handler_params,
                conversation_context=conversation_context,
                span=span,
            )
        is_llm_client_configuration = is_class and issubclass(
            param_annotation, LLMClientConfiguration
        )
        if is_llm_client_configuration:
            if agent_setup is None:
                if has_default:
                    return param.default
                if is_optional:
                    return None
                else:
                    raise ValueError(
                        f"Missing value for required parameter: {param_name}"
                    )
            return agent_setup.llm_client_configuration

        is_span = is_class and issubclass(param_annotation, Span)
        if is_span:
            return span

        # Parse agent configuration
        is_not_annotated = param_annotation == inspect.Parameter.empty
        is_base_model = is_class and issubclass(param_annotation, BaseModel)
        return self.__parse_agent_configuration(
            param_name=param_name,
            has_default=has_default,
            is_optional=is_optional,
            is_not_annotated=is_not_annotated,
            is_base_model=is_base_model,
            param_default=param.default,
            param_annotation=cast(BaseModel, param_annotation),
            handler_params=handler_params,
            agent_setup=agent_setup,
        )

    async def create(
        self,
        agent_identifier: str,
        handler_params: Dict[str, Any],
        agent_name: Optional[str] = None,
        conversation_context: Optional[ConversationContext] = None,
        span: Optional[Span] = None,
        extra_agent_kwargs: Optional[Dict[str, Any]] = None,
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
            )
        agent_cls = await self.__chat_agent_class_registry.get(
            agent_name=agent_name or agent_setup.agent_name
        )
        agent_cls_params = inspect.signature(agent_cls).parameters
        agent_cls_param_values = {
            param_name: await self.__parse_value(
                param=param,
                param_name=param_name,
                handler_params=handler_params,
                conversation_context=conversation_context,
                agent_setup=agent_setup,
                span=span,
            )
            for param_name, param in agent_cls_params.items()
        }
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
                    isinstance(param_value, BaseModel)
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

        if self.__publish_event:
            agent_instance.publish_event = self.__publish_event
        return agent_instance

    async def create_streamable(
        self,
        agent_identifier: str,
        handler_params: Dict[str, Any],
        agent_name: Optional[str] = None,
        conversation_context: Optional[ConversationContext] = None,
        span: Optional[Span] = None,
        extra_agent_kwargs: Optional[Dict[str, Any]] = None,
    ) -> StreamableChatAgent:
        agent_instance = await self.create(
            agent_identifier=agent_identifier,
            agent_name=agent_name,
            handler_params=handler_params,
            conversation_context=conversation_context,
            span=span,
            extra_agent_kwargs=extra_agent_kwargs,
        )
        if not isinstance(agent_instance, StreamableChatAgent):
            raise ValueError(f"Agent {agent_name} is not streamable")

        return agent_instance
