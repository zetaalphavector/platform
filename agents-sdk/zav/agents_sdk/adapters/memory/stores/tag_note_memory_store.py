from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryEntry,
    MemorySaveError,
    MemoryStore,
)
from zav.agents_sdk.adapters.notes.notes_service import NotesService
from zav.agents_sdk.adapters.tags.tags_service import TagsService
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class TagNoteMemoryStoreConfiguration(BaseModel):
    enabled: bool = False
    tag_id: Optional[str] = Field(
        None,
        description="Tag ID where memory entries are stored as tag notes.",
    )
    use_favorites: bool = Field(
        False,
        description=(
            "Use the user's favorites tag for memory storage. "
            "When true, takes precedence over tag_id."
        ),
    )


class TagNoteMemoryStore(MemoryStore):
    source_name = "tag_notes"

    def __init__(
        self,
        notes_service: NotesService,
        tags_service: TagsService,
        enabled: bool,
        tag_id: Optional[str] = None,
        use_favorites: bool = False,
    ) -> None:
        self.enabled = enabled
        self.__tag_id = int(tag_id) if tag_id is not None else None
        self.__use_favorites = use_favorites
        self.__notes_service = notes_service
        self.__tags_service = tags_service

    async def load_context(self) -> List[MemoryEntry]:
        if self.__use_favorites:
            return await self.__load_from_favorites()
        if self.__tag_id is not None:
            return await self.__load_from_tag(self.__tag_id)
        logger.warning("Memory tag ID not configured; no memories will be loaded.")
        return []

    async def __load_from_favorites(self) -> List[MemoryEntry]:
        try:
            notes = await self.__notes_service.retrieve_favorite_tag_notes(
                page=1,
                page_size=300,
            )
        except Exception as e:
            logger.error(f"Failed to load memory from favorites tag: {e}")
            return []
        return self.__notes_to_entries(notes)

    async def __load_from_tag(self, tag_id: int) -> List[MemoryEntry]:
        tag_type = await self.__tags_service.get_tag_type_by_id(tag_id)
        if tag_type is None:
            logger.warning(f"Tag {tag_id} not found or unsupported type; skipping")
            return []
        try:
            notes = await self.__notes_service.retrieve_tag_notes(
                tag_type=tag_type,
                tag_id=tag_id,
                page=1,
                page_size=300,
            )
        except Exception as e:
            logger.error(f"Failed to load memory tag notes for tag {tag_id}: {e}")
            return []
        return self.__notes_to_entries(notes)

    @staticmethod
    def __notes_to_entries(notes: List) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for note in notes:
            content = note.get("content", "")
            if not content:
                continue
            note_id = note.get("id")
            entry_id = str(note_id) if note_id else None
            created_at = note.get("date_created")
            entries.append(
                MemoryEntry(id=entry_id, content=content, created_at=created_at)
            )
        return entries

    async def save(self, content: str) -> MemoryEntry:
        if self.__use_favorites:
            return await self.__save_to_favorites(content)
        if self.__tag_id is not None:
            return await self.__save_to_tag(self.__tag_id, content)
        raise MemorySaveError("Memory tag ID not configured; cannot save memory.")

    async def __save_to_favorites(self, content: str) -> MemoryEntry:
        try:
            result = await self.__notes_service.create_favorite_tag_note(
                content=content
            )
        except Exception as e:
            raise MemorySaveError(f"Failed to save memory to favorites tag: {e}") from e
        note_id = result.get("id")
        entry_id = str(note_id) if note_id else None
        return MemoryEntry(
            id=entry_id,
            content=content,
            created_at=result.get("date_created"),
            metadata={"note_id": str(note_id)} if note_id else None,
        )

    async def __save_to_tag(self, tag_id: int, content: str) -> MemoryEntry:
        tag_type = await self.__tags_service.get_tag_type_by_id(tag_id)
        if tag_type is None:
            raise MemorySaveError(f"Tag {tag_id} not found or unsupported type.")
        try:
            result = await self.__notes_service.create_tag_note(
                tag_type=tag_type,
                tag_id=tag_id,
                content=content,
            )
        except Exception as e:
            raise MemorySaveError(
                f"Failed to save memory tag note for tag {tag_id}: {e}"
            ) from e
        note_id = result.get("id")
        entry_id = str(note_id) if note_id else None
        return MemoryEntry(
            id=entry_id,
            content=content,
            created_at=result.get("date_created"),
            metadata={"note_id": str(note_id)} if note_id else None,
        )


class TagNoteMemoryStoreFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notes_service: NotesService,
        tags_service: TagsService,
        tag_note_memory_store_configuration: TagNoteMemoryStoreConfiguration = (
            TagNoteMemoryStoreConfiguration()
        ),
    ) -> TagNoteMemoryStore:
        return TagNoteMemoryStore(
            notes_service=notes_service,
            tags_service=tags_service,
            tag_id=tag_note_memory_store_configuration.tag_id,
            use_favorites=tag_note_memory_store_configuration.use_favorites,
            enabled=tag_note_memory_store_configuration.enabled,
        )
