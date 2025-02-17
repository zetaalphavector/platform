from typing import Optional, Protocol, Tuple, Type

from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetupRetriever
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistryProtocol


class AgentRegistriesFactory(Protocol):

    async def create(self, tenant: str) -> Tuple[
        AgentSetupRetriever,
        ChatAgentClassRegistryProtocol,
        Optional[Type[AgentDependencyRegistry]],
    ]: ...
