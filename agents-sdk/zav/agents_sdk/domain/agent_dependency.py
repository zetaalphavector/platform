import inspect
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    get_args,
)

from typing_extensions import ParamSpec

from zav.agents_sdk.domain.utils import check_is_optional

T = TypeVar("T")
DEPENDENCY_PARAMS = ParamSpec("DEPENDENCY_PARAMS")


class DependencyGroup(Generic[T]):
    """Marker base for auto-collecting all registered subclasses of a type.

    A concrete subclass declares the base type to collect via
    ``__collects__`` and can then be used as a constructor parameter.
    When ``ChatAgentFactory`` encounters such a parameter, it scans the
    dependency registry for all factories whose return type is a subclass
    of ``__collects__``, resolves each one via normal DI, and wraps them
    in the group.

    Example::

        class ToolsSourceGroup(DependencyGroup[ToolsSource]):
            __collects__ = ToolsSource

    An agent (or another dependency) can then declare
    ``tools_source_group: ToolsSourceGroup`` in its constructor.
    """

    __collects__: Type

    def __init__(self, items: List[T]) -> None:
        self.items = items


class AgentDependencyFactory(ABC, Generic[DEPENDENCY_PARAMS, T]):
    @classmethod
    @abstractmethod
    def create(
        cls, *args: DEPENDENCY_PARAMS.args, **kwargs: DEPENDENCY_PARAMS.kwargs
    ) -> T:
        """Create an instance of the dependency. Arguments could be:
        - llm_client_configuration matched by argument type LLMClientConfiguration
        - a typed object within agent_configuration matched by argument type
        - (un)typed key-value pairs within agent_configuration, matched by argument name
        - (un)typed key-value pairs within the handler command, matched by argument name
        - other dependencies matched by their type
        """
        raise NotImplementedError


class AgentDependencyRegistryProtocol(Protocol):

    def get(
        self, t: type
    ) -> Optional[Union[Type[AgentDependencyFactory], AgentDependencyFactory]]: ...

    def get_subclasses_of(
        self, base: type
    ) -> List[Union[Type[AgentDependencyFactory], AgentDependencyFactory]]: ...


class AgentDependencyRegistry(AgentDependencyRegistryProtocol):
    registry: Dict[
        type, Union[Type[AgentDependencyFactory], AgentDependencyFactory]
    ] = {}

    def __init_subclass__(cls):
        cls.registry = {}

    @classmethod
    def get(
        cls, t: type
    ) -> Optional[Union[Type[AgentDependencyFactory], AgentDependencyFactory]]:
        return cls.registry.get(t)

    @classmethod
    def get_subclasses_of(
        cls, base: type
    ) -> List[Union[Type[AgentDependencyFactory], AgentDependencyFactory]]:
        return [
            factory
            for return_type, factory in cls.registry.items()
            if inspect.isclass(return_type) and issubclass(return_type, base)
        ]

    @classmethod
    def register(
        cls, inst_or_cls: Union[Type[AgentDependencyFactory], AgentDependencyFactory]
    ):
        created_cls = inspect.signature(inst_or_cls.create).return_annotation
        if created_cls == inspect.Signature.empty:
            raise ValueError(
                f"Factory method {inst_or_cls.create} should have a return annotation"
            )

        # Unwrap Optional[X] to X
        if check_is_optional(created_cls):
            created_cls = next(
                arg for arg in get_args(created_cls) if arg is not type(None)
            )

        cls.registry[created_cls] = inst_or_cls
