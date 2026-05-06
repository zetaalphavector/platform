from typing import List

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryEntry,
    MemorySaveError,
    MemoryStore,
    MemoryUpdateError,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class InMemoryMemoryStoreConfiguration(BaseModel):
    enabled: bool = Field(
        False, description="Enable in-memory (ephemeral) memory store."
    )


class InMemoryMemoryStore(MemoryStore):
    source_name = "in_memory"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.__entries: List[MemoryEntry] = []

    async def load_context(self) -> List[MemoryEntry]:
        if not self.enabled:
            return []
        return [
            MemoryEntry(
                id=str(index),
                content=entry.content,
                created_at=entry.created_at,
                metadata=entry.metadata,
            )
            for index, entry in enumerate(self.__entries)
        ]

    async def save(self, content: str) -> MemoryEntry:
        if not self.enabled:
            raise MemorySaveError("In-memory memory store is not enabled.")
        entry = MemoryEntry(content=content)
        self.__entries.append(entry)
        return MemoryEntry(
            id=str(len(self.__entries) - 1),
            content=entry.content,
            created_at=entry.created_at,
            metadata=entry.metadata,
        )

    async def update(self, entry_id: str, content: str) -> MemoryEntry:
        if not self.enabled:
            raise MemoryUpdateError("In-memory memory store is not enabled.")
        for index, entry in enumerate(self.__entries):
            if str(index) == entry_id:
                self.__entries[index] = MemoryEntry(
                    id=entry_id,
                    content=content,
                    created_at=entry.created_at,
                    metadata=entry.metadata,
                )
                return MemoryEntry(
                    id=entry_id,
                    content=content,
                    created_at=entry.created_at,
                    metadata=entry.metadata,
                )
        raise MemoryUpdateError(f"Memory entry '{entry_id}' not found.")


class InMemoryMemoryStoreFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        in_memory_memory_store_configuration: InMemoryMemoryStoreConfiguration = (
            InMemoryMemoryStoreConfiguration()
        ),
    ) -> InMemoryMemoryStore:
        return InMemoryMemoryStore(
            enabled=in_memory_memory_store_configuration.enabled,
        )
