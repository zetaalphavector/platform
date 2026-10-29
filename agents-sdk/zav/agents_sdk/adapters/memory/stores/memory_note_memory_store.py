from typing import Dict, List, Literal, Tuple

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
    source_description = "Memories stored as the user's memory notes."

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

        entries = self.__to_entries(all_notes)
        logger.info(f"Loaded {len(entries)} memory entries from notes")
        return entries

    async def load_recent_context(self, size: int) -> Tuple[List[MemoryEntry], int]:
        self.__permissions.clear()

        all_notes: List = []
        total = 0

        if "own" in self.__include_scopes:
            own_notes, own_total = (
                await self.__notes_service.retrieve_notes_with_total_count(
                    note_object_type="memory",
                    page=1,
                    page_size=size,
                    created_by_me=True,
                    sort_by=["date_created:desc"],
                )
            )
            all_notes.extend(own_notes)
            total += own_total

        if "shared_with_me" in self.__include_scopes:
            shared_notes, shared_total = (
                await self.__notes_service.retrieve_notes_with_total_count(
                    note_object_type="memory",
                    page=1,
                    page_size=size,
                    created_by_me=False,
                    sort_by=["date_created:desc"],
                )
            )
            all_notes.extend(shared_notes)
            total += shared_total

        all_notes.sort(key=lambda note: note.get("date_created") or "", reverse=True)
        all_notes = all_notes[:size]
        entries = self.__to_entries(all_notes)
        logger.info(f"Loaded {len(entries)} memory entries from notes")
        return entries, total

    def __to_entries(self, all_notes: List) -> List[MemoryEntry]:
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
