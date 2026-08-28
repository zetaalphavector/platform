import copy
import inspect
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
    get_args,
)

from zav.llm_tracing import Span
from zav.pydantic_compat import PYDANTIC_V2, BaseModel, _BaseModel

from zav.agents_sdk.domain.agent_creator import AgentCreator
from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyRegistryProtocol,
    DependencyGroup,
)
from zav.agents_sdk.domain.agent_setup_retriever import merge_dicts
from zav.agents_sdk.domain.chat_agent import ChatAgent
from zav.agents_sdk.domain.chat_request import ConversationContext
from zav.agents_sdk.domain.llm_client_factory import LLMClientFactory
from zav.agents_sdk.domain.llm_configuration_store import LLMConfigurationStore
from zav.agents_sdk.domain.utils import (
    check_is_base_model,
    check_is_class,
    check_is_optional,
)
from zav.agents_sdk.security import _PROTECTED_HANDLER_PARAMS, sanitize_bot_params


class AgentCreation(Protocol):
    """How the resolver instantiates agents when a dependency demands one: the
    AgentCreator branch (task/delegate tools) and ChatAgent-typed constructor
    params. ChatAgentFactory is the implementation; a resolver built without
    one can still resolve agent-free dependency trees."""

    async def create_agent(
        self,
        agent_identifier: str,
        handler_params: Dict[str, Any],
        conversation_context: Optional[ConversationContext] = None,
        parent_span: Optional[Span] = None,
        extra_agent_kwargs: Optional[Dict[str, Any]] = None,
        track_resumable: bool = True,
    ) -> ChatAgent: ...


# (dependency_path, instance, is_singleton) — called for every resolved
# dependency while tracking is on; ChatAgentFactory hooks its resumable-state
# bookkeeping here.
DependencyResolvedHook = Callable[[Tuple[str, ...], Any, bool], Awaitable[None]]


@dataclass
class _ResolutionContext:
    """Everything a resolution walk threads through its recursion."""

    configuration: Dict[str, Any]
    handler_params: Dict[str, Any]
    default_agent_identifier: str
    conversation_context: Optional[ConversationContext]
    span: Optional[Span]
    sub_agent_mapping: Optional[Dict[str, str]]
    llm_client_factory: Optional[LLMClientFactory]
    track_resumable: bool
    resolution_cache: Dict[type, Any] = field(default_factory=dict)


class DependencyResolver:
    """The registration-at-start, resolve-per-request engine: resolves a
    class's constructor params from the dependency registry, the configuration
    blocks, and the handler params (protected params win). Agent-agnostic: the
    agent-shaped branches delegate to the injected AgentCreation seam, and
    resumable-state tracking is the caller's hook."""

    def __init__(
        self,
        dependency_registry: Optional[AgentDependencyRegistryProtocol],
        agent_creation: Optional[AgentCreation] = None,
        on_dependency_resolved: Optional[DependencyResolvedHook] = None,
        llm_configuration_store: Optional[LLMConfigurationStore] = None,
    ):
        self.__dependency_registry = dependency_registry
        self.__agent_creation = agent_creation
        self.__on_dependency_resolved = on_dependency_resolved
        self.__llm_configuration_store = llm_configuration_store

    def __llm_client_factory(
        self,
        default_llm_configuration_name: Optional[str],
        allowed_llm_configuration_names: Optional[Sequence[str]],
    ) -> LLMClientFactory:
        return LLMClientFactory(
            llm_configuration_store=self.__llm_configuration_store,
            default_llm_configuration_name=default_llm_configuration_name,
            allowed_llm_configuration_names=allowed_llm_configuration_names,
        )

    async def resolve(
        self,
        target_cls: type,
        configuration: Dict[str, Any],
        handler_params: Dict[str, Any],
        key_prefix: str,
        conversation_context: Optional[ConversationContext] = None,
        span: Optional[Span] = None,
        sub_agent_mapping: Optional[Dict[str, str]] = None,
        default_llm_configuration_name: Optional[str] = None,
        allowed_llm_configuration_names: Optional[Sequence[str]] = None,
        track_resumable: bool = False,
    ) -> Any:
        """Resolve a single dependency (e.g. a provider) from the registry and
        the given configuration blocks."""
        context = _ResolutionContext(
            configuration=configuration,
            handler_params=handler_params,
            default_agent_identifier=key_prefix,
            conversation_context=conversation_context,
            span=span,
            sub_agent_mapping=sub_agent_mapping,
            llm_client_factory=self.__llm_client_factory(
                default_llm_configuration_name, allowed_llm_configuration_names
            ),
            track_resumable=track_resumable,
        )
        param = inspect.Parameter(
            "__resolved_target",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=target_cls,
        )
        return await self.__resolve_param(
            param=param,
            param_name="__resolved_target",
            context=context,
            dependency_path=(key_prefix,),
        )

    async def resolve_parameters(
        self,
        parameters: Mapping[str, inspect.Parameter],
        configuration: Dict[str, Any],
        handler_params: Dict[str, Any],
        key_prefix: str,
        conversation_context: Optional[ConversationContext] = None,
        span: Optional[Span] = None,
        sub_agent_mapping: Optional[Dict[str, str]] = None,
        default_llm_configuration_name: Optional[str] = None,
        allowed_llm_configuration_names: Optional[Sequence[str]] = None,
        track_resumable: bool = True,
    ) -> Dict[str, Any]:
        """Resolve a full constructor signature with one shared resolution
        walk (singletons resolve once across all params)."""
        context = _ResolutionContext(
            configuration=configuration,
            handler_params=handler_params,
            default_agent_identifier=key_prefix,
            conversation_context=conversation_context,
            span=span,
            sub_agent_mapping=sub_agent_mapping,
            llm_client_factory=self.__llm_client_factory(
                default_llm_configuration_name, allowed_llm_configuration_names
            ),
            track_resumable=track_resumable,
        )
        return {
            param_name: await self.__resolve_param(
                param=param,
                param_name=param_name,
                context=context,
                dependency_path=(key_prefix, param_name),
            )
            for param_name, param in parameters.items()
        }

    def __agent_creator(
        self, context: _ResolutionContext, span: Optional[Span]
    ) -> AgentCreator:
        agent_creation = self.__agent_creation

        async def agent_factory(
            factory_agent_identifier: str,
            bot_params: Dict[str, Any],
            factory_conversation_context: Optional[ConversationContext] = None,
            extra_agent_kwargs: Optional[Dict[str, Any]] = None,
        ):
            if agent_creation is None:
                raise ValueError("Agent creation is not available for this resolution.")
            # Agents spawned at runtime through an AgentCreator (the task and
            # delegate tools) run isolated and stateless: a fresh context that
            # returns only a text result. Their resumable dependencies are
            # therefore not tracked or persisted into the parent turn's keyspace,
            # which also stops repeated spawns of the same agent_identifier from
            # colliding on the shared per-factory state key. Durable per-sub-agent
            # resume (each sub-agent owning its own factory and state namespace) is
            # intentionally deferred.
            return await agent_creation.create_agent(
                agent_identifier=factory_agent_identifier,
                handler_params={
                    **context.handler_params,
                    **sanitize_bot_params(bot_params),
                },
                conversation_context=factory_conversation_context,
                parent_span=span,
                extra_agent_kwargs=extra_agent_kwargs,
                track_resumable=False,
            )

        return AgentCreator(
            agent_factory=agent_factory,
            default_agent_identifier=context.default_agent_identifier,
        )

    async def __resolve_sub_agent(
        self,
        has_default: bool,
        is_optional: bool,
        sub_agent_identifier: str,
        param_default: Any,
        context: _ResolutionContext,
    ):
        try:
            if self.__agent_creation is None:
                raise ValueError("Agent creation is not available for this resolution.")
            return await self.__agent_creation.create_agent(
                agent_identifier=sub_agent_identifier,
                handler_params=context.handler_params,
                conversation_context=context.conversation_context,
                parent_span=context.span,
                track_resumable=context.track_resumable,
            )
        except ValueError as e:
            if has_default:
                return param_default
            elif is_optional:
                return None
            else:
                raise e

    def __resolve_configuration(
        self,
        param_name: str,
        has_default: bool,
        is_optional: bool,
        is_not_annotated: bool,
        is_base_model: bool,
        param_default: Any,
        param_annotation: BaseModel,
        context: _ResolutionContext,
    ):
        configuration = context.configuration
        handler_params = context.handler_params
        is_param_missing = param_name not in configuration
        is_protected_handler_param = (
            param_name in _PROTECTED_HANDLER_PARAMS and param_name in handler_params
        )

        if is_protected_handler_param:
            param_value = handler_params[param_name]
        elif is_param_missing:
            if param_name in handler_params:
                param_value = handler_params[param_name]
            elif has_default:
                return param_default
            else:
                raise ValueError(f"Missing value for required parameter: {param_name}")
        else:
            # Configuration found in the setup should take precedence over any
            # runtime overrides (handler_params) to avoid accidental or malicious
            # configuration tampering. Runtime-provided values are only used to
            # fill in missing fields.
            config_value = configuration[param_name]
            param_value = config_value
            if (
                param_name in handler_params
                and isinstance(handler_params[param_name], dict)
                and isinstance(config_value, dict)
            ):
                # Perform a deep merge so nested dictionaries are combined while
                # ensuring configuration values take precedence.
                merged = copy.deepcopy(handler_params[param_name])
                merge_dicts(merged, config_value)
                param_value = merged
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
                if PYDANTIC_V2:
                    return param_annotation.model_validate(param_value)
                else:
                    return param_annotation.parse_obj(param_value)
            elif isinstance(param_value, str):
                if PYDANTIC_V2:
                    return param_annotation.model_validate_json(param_value)
                else:
                    return param_annotation.parse_raw(param_value)
            elif isinstance(param_value, _BaseModel):
                if PYDANTIC_V2:
                    return param_annotation.model_validate(param_value)
                else:
                    return param_annotation.from_orm(param_value)
            else:
                raise ValueError(
                    f"Unsupported type for {param_name}: {type(param_value)}"
                )
        else:
            # We assume this is a value that can be directly passed
            return param_value

    async def __resolve_param(
        self,
        param: inspect.Parameter,
        param_name: str,
        context: _ResolutionContext,
        dependency_path: Tuple[str, ...],
    ) -> Optional[Any]:
        param_annotation = param.annotation
        is_optional = check_is_optional(param_annotation)
        if is_optional:
            param_annotation = next(
                annotation
                for annotation in get_args(param_annotation)
                if annotation is not type(None)  # noqa: E721
            )
        is_class = check_is_class(param_annotation)
        # Parse AgentCreator
        if is_class and issubclass(param_annotation, AgentCreator):
            return self.__agent_creator(context=context, span=context.span)
        # Parse registered dependency
        if self.__dependency_registry and is_class:
            dependency = self.__dependency_registry.get(param_annotation)
            if dependency:
                is_singleton = getattr(dependency, "__singleton__", False)
                if is_singleton and param_annotation in context.resolution_cache:
                    return context.resolution_cache[param_annotation]
                # The dependency needs to be inspected and initialized
                dependency_params = inspect.signature(dependency.create).parameters
                result = dependency.create(
                    **{
                        dep_param_name: (
                            await self.__resolve_param(
                                param=dep_param,
                                param_name=dep_param_name,
                                context=context,
                                dependency_path=(*dependency_path, dep_param_name),
                            )
                        )
                        for dep_param_name, dep_param in dependency_params.items()
                        if dep_param_name != "self"
                    }
                )
                if is_singleton:
                    context.resolution_cache[param_annotation] = result
                if context.track_resumable and self.__on_dependency_resolved:
                    await self.__on_dependency_resolved(
                        dependency_path, result, is_singleton
                    )
                return result
        # Parse dependency group
        if (
            self.__dependency_registry
            and is_class
            and issubclass(param_annotation, DependencyGroup)
            and hasattr(param_annotation, "__collects__")
        ):
            base_type = param_annotation.__collects__
            factories = self.__dependency_registry.get_subclasses_of(base_type)
            items = []
            for factory in factories:
                factory_name = (
                    factory.__name__
                    if inspect.isclass(factory)
                    else factory.__class__.__name__
                )
                factory_params = inspect.signature(factory.create).parameters
                item = factory.create(
                    **{
                        fp_name: (
                            await self.__resolve_param(
                                param=fp,
                                param_name=fp_name,
                                context=context,
                                dependency_path=(
                                    *dependency_path,
                                    factory_name,
                                    fp_name,
                                ),
                            )
                        )
                        for fp_name, fp in factory_params.items()
                        if fp_name != "self"
                    }
                )
                if context.track_resumable and self.__on_dependency_resolved:
                    await self.__on_dependency_resolved(
                        (*dependency_path, factory_name),
                        item,
                        getattr(factory, "__singleton__", False),
                    )
                items.append(item)
            return param_annotation(items=items)
        has_default = param.default != inspect.Parameter.empty
        # Parse conversation context
        is_conversation_context = is_class and issubclass(
            param_annotation, ConversationContext
        )
        if is_conversation_context:
            return context.conversation_context
        # Parse sub agent
        is_chat_agent = is_class and issubclass(param_annotation, ChatAgent)
        if is_chat_agent:
            sub_agent_name = cast(ChatAgent, param_annotation).agent_name
            sub_agent_identifier = sub_agent_name
            if context.sub_agent_mapping:
                sub_agent_identifier = context.sub_agent_mapping.get(
                    sub_agent_name, sub_agent_name
                )
            return await self.__resolve_sub_agent(
                has_default=has_default,
                is_optional=is_optional,
                sub_agent_identifier=sub_agent_identifier,
                param_default=param.default,
                context=context,
            )
        is_llm_client_factory = is_class and issubclass(
            param_annotation, LLMClientFactory
        )
        if is_llm_client_factory:
            return context.llm_client_factory

        is_span = is_class and issubclass(param_annotation, Span)
        if is_span:
            return context.span

        # Parse configuration block
        is_not_annotated = param_annotation == inspect.Parameter.empty
        is_base_model = is_class and check_is_base_model(param_annotation)
        return self.__resolve_configuration(
            param_name=param_name,
            has_default=has_default,
            is_optional=is_optional,
            is_not_annotated=is_not_annotated,
            is_base_model=is_base_model,
            param_default=param.default,
            param_annotation=cast(BaseModel, param_annotation),
            context=context,
        )
