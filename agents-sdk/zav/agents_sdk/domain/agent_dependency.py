import inspect
from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, Protocol, Type, TypeVar, Union

from typing_extensions import ParamSpec

T = TypeVar("T")
DEPENDENCY_PARAMS = ParamSpec("DEPENDENCY_PARAMS")


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
    def register(
        cls, inst_or_cls: Union[Type[AgentDependencyFactory], AgentDependencyFactory]
    ):
        created_cls = inspect.signature(inst_or_cls.create).return_annotation
        if created_cls == inspect.Signature.empty:
            raise ValueError(
                f"Factory method {inst_or_cls.create} should have a return annotation"
            )
        cls.registry[created_cls] = inst_or_cls
