import json
import os
from typing import Any, Dict, List, Optional, Protocol

from zav.llm_domain import LLMClientConfiguration
from zav.logging import logger

from zav.agents_sdk.domain.agent_setup_retriever import merge_dicts


class LLMConfigurationStore(Protocol):
    """Named LLM configurations, scoped to one tenant or one local project."""

    def get(self, name: str) -> Optional[LLMClientConfiguration]: ...

    def names(self) -> List[str]: ...


class StaticLLMConfigurationStore:
    def __init__(self, configurations: Dict[str, LLMClientConfiguration]):
        self.__configurations = configurations

    def get(self, name: str) -> Optional[LLMClientConfiguration]:
        return self.__configurations.get(name)

    def names(self) -> List[str]:
        return sorted(self.__configurations)


class LayeredLLMConfigurationStore:
    """A mutable overlay on a base store: inline agent configurations and
    runtime overrides live in the overlay, named entries in the base."""

    def __init__(self, base: Optional[LLMConfigurationStore] = None):
        self.__base = base
        self.__overrides: Dict[str, LLMClientConfiguration] = {}

    def set(self, name: str, configuration: LLMClientConfiguration) -> None:
        self.__overrides[name] = configuration

    def get(self, name: str) -> Optional[LLMClientConfiguration]:
        configuration = self.__overrides.get(name)
        if configuration is not None:
            return configuration
        if self.__base is not None:
            return self.__base.get(name)
        return None

    def names(self) -> List[str]:
        base_names = self.__base.names() if self.__base is not None else []
        return sorted({*base_names, *self.__overrides})


class LLMConfigurationStoreFromFile:
    """llm_configurations.json plus the secret half in env/, merged by name.
    Paths default to siblings of agent_setups.json."""

    def __init__(
        self,
        agent_setups_path: Optional[str] = None,
        llm_configurations_path: Optional[str] = None,
        secret_llm_configurations_path: Optional[str] = None,
    ):
        if llm_configurations_path is None and agent_setups_path:
            llm_configurations_path = os.path.join(
                os.path.dirname(agent_setups_path), "llm_configurations.json"
            )
        if secret_llm_configurations_path is None and agent_setups_path:
            secret_llm_configurations_path = os.path.join(
                os.path.dirname(agent_setups_path), "env", "llm_configurations.json"
            )
        configurations: Dict[str, Dict[str, Any]] = {}
        if llm_configurations_path and os.path.isfile(llm_configurations_path):
            with open(llm_configurations_path, "r") as file:
                for configuration in json.load(file):
                    name = configuration.get("name")
                    if name:
                        configurations[name] = configuration
        if secret_llm_configurations_path and os.path.isfile(
            secret_llm_configurations_path
        ):
            with open(secret_llm_configurations_path, "r") as file:
                for secret_configuration in json.load(file):
                    name = secret_configuration.get("name")
                    if not name:
                        continue
                    if name in configurations:
                        merge_dicts(configurations[name], secret_configuration)
                    else:
                        configurations[name] = secret_configuration
        self.__configurations = configurations

    def names(self) -> List[str]:
        return sorted(self.__configurations)

    def get(self, name: str) -> Optional[LLMClientConfiguration]:
        configuration = self.__configurations.get(name)
        if configuration is None:
            return None
        try:
            return LLMClientConfiguration.parse_obj(
                {key: value for key, value in configuration.items() if key != "name"}
            )
        except Exception as error:
            logger.warning(
                "LLM configuration '%s' could not be materialized into a "
                "client configuration: %s",
                name,
                error,
            )
            return None
