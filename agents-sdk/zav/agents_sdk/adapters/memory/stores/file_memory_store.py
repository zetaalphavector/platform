import json
from pathlib import Path
from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryEntry,
    MemorySaveError,
    MemoryStore,
    MemoryUpdateError,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class FileMemoryStoreConfiguration(BaseModel):
    enabled: bool = False
    file_path: Optional[str] = Field(
        None, description="Path to the JSON file for persistent memory storage."
    )


class FileMemoryStore(MemoryStore):
    source_name = "file"

    def __init__(self, enabled: bool, path: Optional[str] = None) -> None:
        self.enabled = enabled
        self.__path = Path(path) if path else None
        if self.__path:
            self.__path.parent.mkdir(parents=True, exist_ok=True)

    async def load_context(self) -> List[MemoryEntry]:
        if not self.__path or not self.__path.exists():
            return []
        try:
            data = json.loads(self.__path.read_text())
            entries = []
            for index, entry_data in enumerate(data):
                entries.append(
                    MemoryEntry(
                        **{
                            **entry_data,
                            "id": str(index),
                        }
                    )
                )
            return entries
        except Exception as e:
            logger.error(f"Failed to read memory file {self.__path}: {e}")
            return []

    async def save(self, content: str) -> MemoryEntry:
        if not self.__path:
            raise MemorySaveError("File memory store is not configured.")
        entries = await self.load_context()
        entry = MemoryEntry(id=str(len(entries)), content=content)
        entries.append(entry)
        self.__persist(entries)
        return entry

    async def update(self, entry_id: str, content: str) -> MemoryEntry:
        if not self.__path:
            raise MemoryUpdateError("File memory store is not configured.")
        entries = await self.load_context()
        for i, entry in enumerate(entries):
            if entry.id == entry_id:
                entries[i] = MemoryEntry(
                    id=entry_id,
                    content=content,
                    created_at=entry.created_at,
                    metadata=entry.metadata,
                )
                self.__persist(entries)
                return entries[i]
        raise MemoryUpdateError(f"Memory entry '{entry_id}' not found.")

    def __persist(self, entries: List[MemoryEntry]) -> None:
        if not self.__path:
            return
        try:
            self.__path.write_text(
                json.dumps(
                    [e.model_dump() for e in entries],
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception as e:
            raise MemorySaveError(
                f"Failed to write memory file {self.__path}: {e}"
            ) from e


class FileMemoryStoreFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        file_memory_store_configuration: FileMemoryStoreConfiguration = (
            FileMemoryStoreConfiguration()
        ),
    ) -> FileMemoryStore:
        return FileMemoryStore(
            path=file_memory_store_configuration.file_path,
            enabled=file_memory_store_configuration.enabled,
        )
