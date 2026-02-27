from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.memory.memory_store import MemoryEntry, MemoryStore
from zav.agents_sdk.adapters.notes.notes_service import NotesService
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class TagNoteMemoryStoreConfig(BaseModel):
    tag_id: Optional[str] = Field(
        None,
        description="Tag ID where memory entries are stored as tag notes.",
    )


class TagNoteMemoryStore(MemoryStore):
    source_name = "tag_notes"

    def __init__(self, tag_id: Optional[str], notes_service: NotesService) -> None:
        self.__tag_id = int(tag_id) if tag_id is not None else None
        self.__notes_service = notes_service

    async def load_context(self) -> List[MemoryEntry]:
        if self.__tag_id is None:
            logger.warning("Memory tag ID not configured; no memories will be loaded.")
            return []
        try:
            notes = await self.__notes_service.retrieve_tag_notes(
                tag_type="own",
                tag_id=self.__tag_id,
                page=1,
                page_size=300,
            )
        except Exception as e:
            logger.error(
                f"Failed to load memory tag notes for tag {self.__tag_id}: {e}"
            )
            return []

        entries: List[MemoryEntry] = []
        for note in notes:
            content = note.get("content", "")
            if not content:
                continue
            entries.append(MemoryEntry(content=content))
        return entries

    async def save(self, content: str) -> str:
        if self.__tag_id is None:
            logger.warning("Memory tag ID not configured; cannot save memory.")
            return "Error saving to memory: no memory tag ID configured."
        try:
            await self.__notes_service.create_tag_note(
                tag_type="own",
                tag_id=self.__tag_id,
                content=content,
            )
        except Exception as e:
            logger.error(f"Failed to save memory tag note for tag {self.__tag_id}: {e}")
            return f"Error saving to memory: {e}"
        return f"Saved to memory: {content}"


class TagNoteMemoryStoreFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notes_service: NotesService,
        tag_note_memory_store_config: TagNoteMemoryStoreConfig = (
            TagNoteMemoryStoreConfig()
        ),
    ) -> TagNoteMemoryStore:
        return TagNoteMemoryStore(
            tag_id=tag_note_memory_store_config.tag_id,
            notes_service=notes_service,
        )
