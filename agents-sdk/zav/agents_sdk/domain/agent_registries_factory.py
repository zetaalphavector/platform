from typing import Optional, Protocol, Tuple

from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistryProtocol
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetupRetriever
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistryProtocol
from zav.agents_sdk.domain.llm_configuration_store import LLMConfigurationStore
from zav.agents_sdk.domain.request_headers import RequestHeaders
from zav.agents_sdk.domain.secret_store import SecretStore


class AgentRegistriesFactory(Protocol):

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
    ]: ...
