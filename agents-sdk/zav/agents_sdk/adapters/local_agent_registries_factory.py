from typing import Optional, Tuple, Type

from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetupRetriever
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistryProtocol


class LocalAgentRegistriesFactory(AgentRegistriesFactory):
    def __init__(
        self,
        agent_setup_retriever: AgentSetupRetriever,
        chat_agent_class_registry: ChatAgentClassRegistryProtocol,
        agent_dependency_registry: Optional[Type[AgentDependencyRegistry]] = None,
    ):
        self.__agent_setup_retriever = agent_setup_retriever
        self.__chat_agent_class_registry = chat_agent_class_registry
        self.__agent_dependency_registry = agent_dependency_registry

    async def create(self, tenant: str) -> Tuple[
        AgentSetupRetriever,
        ChatAgentClassRegistryProtocol,
        Optional[Type[AgentDependencyRegistry]],
    ]:
        return (
            self.__agent_setup_retriever,
            self.__chat_agent_class_registry,
            self.__agent_dependency_registry,
        )
