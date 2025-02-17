from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from zav.llm_domain import LLMClientConfiguration
from zav.llm_tracing import TracingConfiguration


def merge_dicts(d1, d2):
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            merge_dicts(d1[key], value)
        else:
            d1[key] = value


class AgentSetup(BaseModel):
    agent_identifier: str
    agent_name: str
    llm_client_configuration: Optional[LLMClientConfiguration] = None
    agent_configuration: Optional[Dict[str, Any]] = None
    sub_agent_mapping: Optional[Dict[str, str]] = None
    tracing_configuration: Optional[TracingConfiguration] = None

    def patch(self, update: Optional[Dict] = None):
        """Creates a new instance of AgentSetup with the updated values."""
        if not update:
            return self

        current = self.dict()
        merge_dicts(current, update)
        return AgentSetup.parse_obj(current)


class AgentSetupRetriever(ABC):
    @abstractmethod
    async def get(self, agent_identifier: str) -> Optional[AgentSetup]:
        raise NotImplementedError

    @abstractmethod
    async def list(self) -> List[AgentSetup]:
        raise NotImplementedError
