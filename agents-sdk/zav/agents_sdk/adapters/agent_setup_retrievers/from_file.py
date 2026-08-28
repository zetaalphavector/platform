import json
from typing import Any, Dict, Optional

from zav.llm_domain import LLMClientConfiguration
from zav.logging import logger

from zav.agents_sdk.adapters.agent_setup_retrievers.from_local import (
    LocalAgentSetupRetriever,
)
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetup, merge_dicts
from zav.agents_sdk.domain.llm_configuration_store import (
    LayeredLLMConfigurationStore,
    LLMConfigurationStore,
    LLMConfigurationStoreFromFile,
)


def _inline_llm_configuration_name(agent_identifier: str) -> str:
    return f"__agent__:{agent_identifier}"


class AgentSetupRetrieverFromFile(LocalAgentSetupRetriever):
    def __init__(
        self,
        file_path: Optional[str] = None,
        secret_file_path: Optional[str] = None,
        llm_configurations_path: Optional[str] = None,
        secret_llm_configurations_path: Optional[str] = None,
        llm_configuration_store: Optional[LLMConfigurationStore] = None,
    ):
        raw_setups: Dict[str, Dict[str, Any]] = {}
        if file_path:
            with open(file_path, "r") as file:
                for raw_setup in json.load(file):
                    raw_setups[raw_setup["agent_identifier"]] = raw_setup
        if secret_file_path:
            with open(secret_file_path, "r") as file:
                for secret_setup in json.load(file):
                    agent_identifier = secret_setup["agent_identifier"]
                    if agent_identifier in raw_setups:
                        merge_dicts(raw_setups[agent_identifier], secret_setup)

        base_store = (
            llm_configuration_store
            if llm_configuration_store is not None
            else LLMConfigurationStoreFromFile(
                agent_setups_path=file_path,
                llm_configurations_path=llm_configurations_path,
                secret_llm_configurations_path=secret_llm_configurations_path,
            )
        )
        store = LayeredLLMConfigurationStore(base=base_store)

        agent_setups = []
        for agent_identifier, raw_setup in raw_setups.items():
            # An inline llm_client_configuration is a file-format convenience:
            # it registers in the store under a reserved name, so the setup
            # carries names only. Inline wins over a named reference.
            inline = raw_setup.pop("llm_client_configuration", None)
            if inline is not None:
                name = _inline_llm_configuration_name(agent_identifier)
                store.set(name, LLMClientConfiguration.parse_obj(inline))
                raw_setup["llm_configuration_name"] = name
            agent_setups.append(AgentSetup.parse_obj(raw_setup))

        self.llm_configuration_store: LayeredLLMConfigurationStore = store
        logger.info(f"Loaded {len(agent_setups)} agent setups from file {file_path}")
        super().__init__(agent_setups)
