from typing import Dict, List, Optional

from zav.agents_sdk.domain.agent_setup_retriever import AgentSetup, AgentSetupRetriever


class LocalAgentSetupRetriever(AgentSetupRetriever):
    def __init__(self, agent_setups: List[AgentSetup]):
        self.__agent_setup_map: Dict[str, AgentSetup] = {
            agent_setup.agent_identifier: agent_setup
            for agent_setup in (agent_setups or [])
        }

    def update_agent_setup(self, agent_identifier: str, agent_setup_patch: Dict):
        self.__agent_setup_map[agent_identifier] = self.__agent_setup_map[
            agent_identifier
        ].patch(agent_setup_patch)

    async def get(self, agent_identifier: str) -> Optional[AgentSetup]:
        return self.__agent_setup_map.get(agent_identifier, None)

    async def list(self) -> List[AgentSetup]:
        return list(self.__agent_setup_map.values())
