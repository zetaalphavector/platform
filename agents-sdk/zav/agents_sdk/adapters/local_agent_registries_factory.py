from typing import Optional, Tuple

from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistryProtocol
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetupRetriever
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistryProtocol
from zav.agents_sdk.domain.llm_configuration_store import LLMConfigurationStore
from zav.agents_sdk.domain.request_headers import RequestHeaders
from zav.agents_sdk.domain.secret_store import SecretStore


class LocalAgentRegistriesFactory(AgentRegistriesFactory):
    def __init__(
        self,
        agent_setup_retriever: AgentSetupRetriever,
        chat_agent_class_registry: ChatAgentClassRegistryProtocol,
        agent_dependency_registry: Optional[AgentDependencyRegistryProtocol] = None,
        llm_configuration_store: Optional[LLMConfigurationStore] = None,
        secret_store: Optional[SecretStore] = None,
    ):
        self.__agent_setup_retriever = agent_setup_retriever
        self.__chat_agent_class_registry = chat_agent_class_registry
        self.__agent_dependency_registry = agent_dependency_registry
        self.__llm_configuration_store = llm_configuration_store
        self.__secret_store = secret_store

    async def create(
        self,
        tenant: str,
        request_headers: Optional[RequestHeaders] = None,
    ) -> Tuple[
        AgentSetupRetriever,
        ChatAgentClassRegistryProtocol,
        Optional[AgentDependencyRegistryProtocol],
        Optional[LLMConfigurationStore],
        Optional[SecretStore],
    ]:
        return (
            self.__agent_setup_retriever,
            self.__chat_agent_class_registry,
            self.__agent_dependency_registry,
            self.__llm_configuration_store,
            self.__secret_store,
        )
