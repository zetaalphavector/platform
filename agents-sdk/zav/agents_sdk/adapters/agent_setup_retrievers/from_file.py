import json
from typing import Dict, Optional

from zav.logging import logger

from zav.agents_sdk.adapters.agent_setup_retrievers.from_local import (
    LocalAgentSetupRetriever,
)
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetup


class AgentSetupRetrieverFromFile(LocalAgentSetupRetriever):
    def __init__(
        self, file_path: Optional[str] = None, secret_file_path: Optional[str] = None
    ):
        agent_setups: Dict[str, AgentSetup] = {}
        if file_path:
            with open(file_path, "r") as file:
                for agent_setup in json.load(file):
                    agent_setups.update(
                        {
                            agent_setup["agent_identifier"]: AgentSetup.parse_obj(
                                agent_setup
                            )
                        }
                    )
        secret_agent_setups_dict: Dict[str, Dict] = {}
        if secret_file_path:
            with open(secret_file_path, "r") as file:
                for secret_agent_setups in json.load(file):
                    secret_agent_setups_dict.update(
                        {secret_agent_setups["agent_identifier"]: secret_agent_setups}
                    )

        for agent_identifier in agent_setups:
            if agent_identifier in secret_agent_setups_dict:
                agent_setups[agent_identifier] = agent_setups[agent_identifier].patch(
                    secret_agent_setups_dict[agent_identifier]
                )

        logger.info(f"Loaded {len(agent_setups)} agent setups from file {file_path}")
        super().__init__(list(agent_setups.values()))
