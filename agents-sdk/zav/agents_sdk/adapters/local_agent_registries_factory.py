from typing import Optional, Tuple

from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistryProtocol
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetupRetriever
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistryProtocol


class LocalAgentRegistriesFactory(AgentRegistriesFactory):
    def __init__(
        self,
        agent_setup_retriever: AgentSetupRetriever,
        chat_agent_class_registry: ChatAgentClassRegistryProtocol,
        agent_dependency_registry: Optional[AgentDependencyRegistryProtocol] = None,
    ):
        self.__agent_setup_retriever = agent_setup_retriever
        self.__chat_agent_class_registry = chat_agent_class_registry
        self.__agent_dependency_registry = agent_dependency_registry

    async def create(self, tenant: str) -> Tuple[
        AgentSetupRetriever,
        ChatAgentClassRegistryProtocol,
        Optional[AgentDependencyRegistryProtocol],
    ]:
        return (
            self.__agent_setup_retriever,
            self.__chat_agent_class_registry,
            self.__agent_dependency_registry,
        )
