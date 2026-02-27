from typing import List

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.memory.memory_store import MemoryEntry, MemoryStore
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class InMemoryMemoryStoreConfig(BaseModel):
    enabled: bool = Field(
        False, description="Enable in-memory (ephemeral) memory store."
    )


class InMemoryMemoryStore(MemoryStore):
    source_name = "in_memory"

    def __init__(self, enabled: bool = False) -> None:
        self.__enabled = enabled
        self.__entries: List[MemoryEntry] = []

    async def load_context(self) -> List[MemoryEntry]:
        if not self.__enabled:
            return []
        return list(self.__entries)

    async def save(self, content: str) -> str:
        if not self.__enabled:
            return "In-memory memory store is not enabled"
        entry = MemoryEntry(content=content)
        self.__entries.append(entry)
        return f"Saved to memory: {content}"


class InMemoryMemoryStoreFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        in_memory_memory_store_config: InMemoryMemoryStoreConfig = (
            InMemoryMemoryStoreConfig()
        ),
    ) -> InMemoryMemoryStore:
        return InMemoryMemoryStore(
            enabled=in_memory_memory_store_config.enabled,
        )
