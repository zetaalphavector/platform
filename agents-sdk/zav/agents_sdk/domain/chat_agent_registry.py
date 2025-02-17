from typing import Callable, Dict, Protocol, Type

from zav.agents_sdk.domain.chat_agent import ChatAgent


class ChatAgentClassRegistryProtocol(Protocol):

    async def get(self, agent_name: str) -> Type[ChatAgent]: ...


class ChatAgentClassRegistry(ChatAgentClassRegistryProtocol):
    registry: Dict[str, Type[ChatAgent]] = {}

    def __init_subclass__(cls):
        cls.registry = {}

    @classmethod
    def register(cls) -> Callable:
        def inner_wrapper(
            wrapped_class: Type[ChatAgent],
        ) -> Type[ChatAgent]:
            cls.registry[wrapped_class.agent_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    async def get(cls, agent_name: str) -> Type[ChatAgent]:
        if agent_name not in cls.registry:
            raise ValueError(f"Unknown agent: {agent_name}")

        return cls.registry[agent_name]
