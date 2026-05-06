from typing import Dict, List, Literal

from zav.logging import logger
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryEntry,
    MemorySaveError,
    MemoryStore,
    MemoryUpdateError,
)
from zav.agents_sdk.adapters.notes.notes_service import NotesService
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

MemoryNoteScope = Literal["own", "shared_with_me"]


class MemoryNoteMemoryStoreConfiguration(BaseModel):
    enabled: bool = False
    include_scopes: List[MemoryNoteScope] = ["own"]


class MemoryNoteMemoryStore(MemoryStore):
    source_name = "memory_notes"

    def __init__(
        self,
        notes_service: NotesService,
        enabled: bool,
        include_scopes: List[MemoryNoteScope] = ["own"],
    ) -> None:
        self.enabled = enabled
        self.__notes_service = notes_service
        self.__include_scopes = include_scopes
        self.__permissions: Dict[str, str] = {}

    async def load_context(self) -> List[MemoryEntry]:
        self.__permissions.clear()

        all_notes: List = []

        if "own" in self.__include_scopes:
            own_notes = await self.__notes_service.retrieve_notes(
                note_object_type="memory",
                page=1,
                page_size=100,
                created_by_me=True,
            )
            all_notes.extend(own_notes)

        if "shared_with_me" in self.__include_scopes:
            shared_notes = await self.__notes_service.retrieve_notes(
                note_object_type="memory",
                page=1,
                page_size=100,
                created_by_me=False,
            )
            all_notes.extend(shared_notes)

        seen_contents: set = set()
        entries: List[MemoryEntry] = []

        for note in all_notes:
            content = note.get("content", "")
            if not content:
                continue
            normalized = content.strip()
            if normalized in seen_contents:
                continue
            seen_contents.add(normalized)

            note_id = note.get("id")
            entry_id = str(note_id) if note_id else None

            permission = note.get("permission")
            if entry_id and permission:
                self.__permissions[entry_id] = str(permission)

            entries.append(
                MemoryEntry(
                    id=entry_id,
                    content=content,
                    created_at=note.get("date_created"),
                )
            )

        logger.info(f"Loaded {len(entries)} memory entries from notes")
        return entries

    async def save(self, content: str) -> MemoryEntry:
        try:
            result = await self.__notes_service.create_note(
                content=content, object_type="memory"
            )
        except Exception as e:
            raise MemorySaveError(f"Failed to save memory note: {e}") from e
        note_id = result.get("id")
        entry_id = str(note_id) if note_id else None
        return MemoryEntry(
            id=entry_id,
            content=content,
            created_at=result.get("date_created"),
            metadata={"note_id": str(note_id)} if note_id else None,
        )

    async def update(self, entry_id: str, content: str) -> MemoryEntry:
        permission = self.__permissions.get(entry_id, "").upper()
        if permission == "READ":
            raise MemoryUpdateError(
                f"Memory entry '{entry_id}' is read-only — you don't have "
                f"permission to edit it."
            )

        try:
            note_id = int(entry_id)
        except (ValueError, TypeError):
            raise MemoryUpdateError(
                f"Memory entry '{entry_id}' has an invalid identifier."
            )

        try:
            await self.__notes_service.update_note(
                note_id=note_id,
                content=content,
                object_type="memory",
            )
        except Exception as e:
            raise MemoryUpdateError(
                f"Failed to update memory entry '{entry_id}': {e}"
            ) from e

        return MemoryEntry(
            id=entry_id,
            content=content,
            metadata={"note_id": entry_id},
        )


class MemoryNoteMemoryStoreFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notes_service: NotesService,
        memory_note_memory_store_configuration: MemoryNoteMemoryStoreConfiguration = (
            MemoryNoteMemoryStoreConfiguration()
        ),
    ) -> MemoryNoteMemoryStore:
        return MemoryNoteMemoryStore(
            notes_service=notes_service,
            enabled=memory_note_memory_store_configuration.enabled,
            include_scopes=memory_note_memory_store_configuration.include_scopes,
        )
