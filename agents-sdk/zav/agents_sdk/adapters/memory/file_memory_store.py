import json
from pathlib import Path
from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.memory.memory_store import MemoryEntry, MemoryStore
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class FileMemoryStoreConfig(BaseModel):
    file_path: Optional[str] = Field(
        None, description="Path to the JSON file for persistent memory storage."
    )


class FileMemoryStore(MemoryStore):
    source_name = "file"

    def __init__(self, path: Optional[str] = None) -> None:
        self.__path = Path(path) if path else None
        if self.__path:
            self.__path.parent.mkdir(parents=True, exist_ok=True)

    async def load_context(self) -> List[MemoryEntry]:
        if not self.__path or not self.__path.exists():
            return []
        try:
            data = json.loads(self.__path.read_text())
            return [MemoryEntry(**entry) for entry in data]
        except Exception as e:
            logger.error(f"Failed to read memory file {self.__path}: {e}")
            return []

    async def save(self, content: str) -> str:
        if not self.__path:
            return "File memory store is not configured"
        entries = await self.load_context()
        entries.append(MemoryEntry(content=content))
        try:
            self.__path.write_text(
                json.dumps(
                    [e.dict() for e in entries],
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception as e:
            logger.error(f"Failed to write memory file {self.__path}: {e}")
            return f"Error saving to memory: {e}"
        return f"Saved to memory: {content}"


class FileMemoryStoreFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        file_memory_store_config: FileMemoryStoreConfig = FileMemoryStoreConfig(),
    ) -> FileMemoryStore:
        return FileMemoryStore(file_memory_store_config.file_path)
