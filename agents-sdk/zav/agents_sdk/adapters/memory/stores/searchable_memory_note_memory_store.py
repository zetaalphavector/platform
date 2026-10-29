import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import yaml
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryEntry,
    MemoryReadError,
    MemorySaveError,
    MemoryUpdateError,
    RetrievalMethod,
    SearchableMemoryStore,
)
from zav.agents_sdk.adapters.notes.notes_service import NotesService
from zav.agents_sdk.adapters.tools.sources.index_tools_source import IndexToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

MemoryNoteScope = Literal["own", "shared_with_me"]

# Chat provenance is kept as a YAML frontmatter block at the top of the note
# content, under `memory_origins`: a list of {session_id, at} for
# the chats the memory was created and edited from. The store writes and strips
# it, so the agent never sees it; the frontend parses it to show the originating chats.
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
# The search index highlight keeps the frontmatter text but drops its fence lines.
_UNFENCED_ORIGINS_RE = re.compile(r"\Amemory_origins:\n(?:[-\s].*\n)*")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RetrievedMemoryEntry(BaseModel):
    id: Optional[str] = Field(
        None,
        description=(
            "Memory entry identifier. Pass it to the read-memory tool to fetch "
            "the full memory content."
        ),
    )
    snippet: str = Field(
        ...,
        description="The part of the memory that matched the search query.",
    )
    created_at: Optional[str] = Field(
        None, description="ISO date when the memory was created."
    )
    source_url: Optional[str] = Field(None, description="Link to cite this memory.")


class SearchableMemoryNoteMemoryStoreConfiguration(BaseModel):
    enabled: bool = False
    include_scopes: List[MemoryNoteScope] = ["own"]
    max_preloaded_memories: int = 100
    frontmatter_enabled: bool = True
    search_id_field: str = "custom_metadata.metadata.identifier"
    search_snippet_field: str = "highlight"
    search_created_at_field: str = "custom_metadata.metadata.created"


class SearchableMemoryNoteMemoryStore(SearchableMemoryStore):
    source_name = "searchable_memory_notes"
    source_description = "Memory notes the agent can search."

    def __init__(
        self,
        notes_service: NotesService,
        index_tools: IndexToolsSource,
        enabled: bool,
        include_scopes: List[MemoryNoteScope] = ["own"],
        max_preloaded_memories: int = 100,
        frontmatter_enabled: bool = True,
        search_id_field: str = "custom_metadata.metadata.identifier",
        search_snippet_field: str = "highlight",
        search_created_at_field: str = "custom_metadata.metadata.created",
        session_id: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.__notes_service = notes_service
        self.__index_tools = index_tools
        self.__include_scopes = include_scopes
        self.__frontmatter_enabled = frontmatter_enabled
        self.__search_id_field = search_id_field
        self.__search_snippet_field = search_snippet_field
        self.__search_created_at_field = search_created_at_field
        self.__session_id = session_id
        self.__permissions: Dict[str, str] = {}
        self.__max_preloaded_memories = max_preloaded_memories

    @staticmethod
    def __note_url(note_id: Any) -> str:
        return f"/notes/{note_id}"

    @staticmethod
    def __split_origins(content: str) -> Tuple[str, List[Dict[str, Any]]]:
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return content, []
        try:
            front = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return content, []
        origins = front.get("memory_origins") if isinstance(front, dict) else None
        body = content[match.end() :].lstrip("\n")
        return body, origins if isinstance(origins, list) else []

    @staticmethod
    def __with_origins(body: str, origins: List[Dict[str, Any]]) -> str:
        if not origins:
            return body
        front = yaml.safe_dump({"memory_origins": origins}, sort_keys=False).strip()
        return f"---\n{front}\n---\n\n{body}"

    async def __existing_origins(self, note_id: int) -> List[Dict[str, Any]]:
        try:
            existing = await self.__notes_service.retrieve_note(note_id=note_id)
        except Exception:
            return []
        _, origins = self.__split_origins(
            existing.get("content", "") if existing else ""
        )
        return origins

    def __add_current_origin(
        self, origins: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.__session_id:
            return origins
        if any(o.get("session_id") == self.__session_id for o in origins):
            return origins
        return [*origins, {"session_id": self.__session_id, "at": _utc_now_iso()}]

    async def load_context(self) -> List[MemoryEntry]:
        entries, _ = await self.__load(self.__max_preloaded_memories)
        return entries

    async def load_recent_context(self, size: int) -> Tuple[List[MemoryEntry], int]:
        return await self.__load(size)

    async def __load(self, size: int) -> Tuple[List[MemoryEntry], int]:
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
            content, _ = self.__split_origins(note.get("content", ""))
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
                    source_url=self.__note_url(note_id) if note_id else None,
                )
            )

        return entries

    async def save(self, content: str) -> MemoryEntry:
        origins = self.__add_current_origin([]) if self.__frontmatter_enabled else []
        try:
            result = await self.__notes_service.create_note(
                content=self.__with_origins(content, origins),
                object_type="memory",
            )
        except Exception as e:
            raise MemorySaveError(f"Failed to save memory note: {e}") from e
        note_id = result.get("id")
        entry_id = str(note_id) if note_id else None
        return MemoryEntry(
            id=entry_id,
            content=content,
            created_at=result.get("date_created"),
            source_url=self.__note_url(note_id) if note_id else None,
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

        # Preserve the chats the memory already came from and add this one; the
        # Agent only ever sends the body, so the store owns the provenance.
        origins = (
            self.__add_current_origin(await self.__existing_origins(note_id))
            if self.__frontmatter_enabled
            else []
        )

        try:
            await self.__notes_service.update_note(
                note_id=note_id,
                content=self.__with_origins(content, origins),
            )
        except Exception as e:
            raise MemoryUpdateError(
                f"Failed to update memory entry '{entry_id}': {e}"
            ) from e

        return MemoryEntry(
            id=entry_id,
            content=content,
            source_url=self.__note_url(note_id),
            metadata={"note_id": entry_id},
        )

    async def read(self, entry_id: str) -> MemoryEntry:
        try:
            note_id = int(entry_id)
        except (ValueError, TypeError):
            raise MemoryReadError(
                f"Memory entry '{entry_id}' has an invalid identifier."
            )

        try:
            note = await self.__notes_service.retrieve_note(note_id=note_id)
        except Exception as e:
            raise MemoryReadError(
                f"Failed to read memory entry '{entry_id}': {e}"
            ) from e

        if not note:
            raise MemoryReadError(f"Memory entry '{entry_id}' was not found.")

        note_id_value = note.get("id")
        content, _ = self.__split_origins(note.get("content", ""))
        return MemoryEntry(
            id=str(note_id_value) if note_id_value else entry_id,
            content=content,
            created_at=note.get("date_created"),
            source_url=(
                self.__note_url(note_id_value)
                if note_id_value
                else self.__note_url(entry_id)
            ),
        )

    async def search(
        self,
        query: str,
        retrieval_method: Optional[str] = None,
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = await self.__index_tools.search_notes(
            query=query,
            filters={"metadata.DCMI.format": "memory_note"},
            retrieval_method=cast(Optional[RetrievalMethod], retrieval_method),
            page=page,
            page_size=page_size,
            metadata_fields=[
                self.__search_id_field,
                self.__search_snippet_field,
                self.__search_created_at_field,
            ],
        )
        entries = self.__hits_to_retrieved(result.get("results", []))
        return {
            "memories": [e.model_dump(exclude_none=True) for e in entries],
            "total_hits": result.get("total_hits", 0),
            "page": result.get("page", page),
        }

    def __hits_to_retrieved(
        self,
        hits: List[Dict[str, Any]],
    ) -> List[RetrievedMemoryEntry]:
        # search_notes keys each returned metadata value by its field's leaf name.
        id_key = self.__search_id_field.rsplit(".", 1)[-1]
        snippet_key = self.__search_snippet_field.rsplit(".", 1)[-1]
        created_key = self.__search_created_at_field.rsplit(".", 1)[-1]
        entries: List[RetrievedMemoryEntry] = []
        for hit in hits:
            metadata = hit.get("metadata", {})
            snippet, _ = self.__split_origins(metadata.get(snippet_key) or "")
            snippet = _UNFENCED_ORIGINS_RE.sub("", snippet).strip()
            if not snippet:
                continue
            note_id = metadata.get(id_key)
            if not note_id:
                continue
            entries.append(
                RetrievedMemoryEntry(
                    id=str(note_id) if note_id else None,
                    snippet=snippet,
                    created_at=metadata.get(created_key),
                    source_url=self.__note_url(note_id) if note_id else None,
                )
            )
        return entries


class SearchableMemoryNoteMemoryStoreFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notes_service: NotesService,
        index_tools: IndexToolsSource,
        searchable_memory_note_memory_store_configuration: (
            SearchableMemoryNoteMemoryStoreConfiguration
        ) = SearchableMemoryNoteMemoryStoreConfiguration(),
        session_id: Optional[str] = None,
    ) -> SearchableMemoryNoteMemoryStore:
        config = searchable_memory_note_memory_store_configuration
        return SearchableMemoryNoteMemoryStore(
            notes_service=notes_service,
            index_tools=index_tools,
            enabled=config.enabled,
            include_scopes=config.include_scopes,
            max_preloaded_memories=config.max_preloaded_memories,
            frontmatter_enabled=config.frontmatter_enabled,
            search_id_field=config.search_id_field,
            search_snippet_field=config.search_snippet_field,
            search_created_at_field=config.search_created_at_field,
            session_id=session_id,
        )
