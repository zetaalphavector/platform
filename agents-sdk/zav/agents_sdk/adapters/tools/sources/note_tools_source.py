from typing import Any, Dict, List, Optional, Union

from zav.logging import logger

from zav.agents_sdk.adapters.notes.notes_service import NotesService
from zav.agents_sdk.adapters.tools.sources.index_tools_source import IndexToolsSource
from zav.agents_sdk.adapters.tools.tools_source import (
    ToolsSource,
    ToolsSourceConfiguration,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable

# The hosted chat service saves every conversation as a free note that starts
# with this.
_CHAT_NOTE_PREFIX = "# Chat\n"


class NoteToolsSourceConfiguration(ToolsSourceConfiguration):
    enabled: bool = False
    list_page_size: int = 20
    search_filters: Dict[str, Union[str, List[str]]] = {
        "metadata.DCMI.format": "free_note"
    }
    search_id_field: str = "custom_metadata.metadata.identifier"
    search_content_field: str = "custom_metadata.metadata.abstract"
    search_created_at_field: str = "custom_metadata.metadata.created"
    exclude_chat_notes: bool = True


class NoteToolsSource(ToolsSource):

    source_name = "note_tools"
    source_description = "Read and write the user's notes."

    def __init__(
        self,
        notes_service: NotesService,
        note_tools_source_configuration: NoteToolsSourceConfiguration,
        index_tools: Optional[IndexToolsSource] = None,
    ):
        self.__notes_service = notes_service
        self.__config = note_tools_source_configuration
        self.__index_tools = index_tools
        self.defer_tools = note_tools_source_configuration.defer_tools
        self.enabled = note_tools_source_configuration.enabled

    async def get_tools(self) -> List[Tool]:
        tools = [
            Tool.from_callable(
                name="create_note",
                executable=self.create_note,
            ),
            Tool.from_callable(
                name="create_document_note",
                executable=self.create_document_note,
            ),
            Tool.from_callable(
                name="update_note",
                executable=self.update_note,
            ),
            Tool.from_callable(
                name="delete_note",
                executable=self.delete_note,
            ),
            Tool.from_callable(
                name="list_my_notes",
                executable=self.list_my_notes,
            ),
        ]
        if self.__index_tools is not None:
            tools.append(
                Tool.from_callable(
                    name="search_notes",
                    executable=self.search_notes,
                )
            )
        return tools

    @streamable(
        running_text="Saving note...",
        completed_text="Note saved.",
        params_transform=hide,
        response_transform=hide,
    )
    async def create_note(self, content: str) -> Dict[str, Any]:
        """Create a standalone personal note.

        Use this to save a free-form note for the user (a takeaway, a
        reminder, a summary). The note is private to the user.

        Args:
            content: The note text (markdown supported).

        Returns:
            Dict with 'note_id' (pass it to tag_note to tag this note) and
            the created note.
        """
        note = await self.__notes_service.create_note(content=content)
        return {"status": "created", "note_id": note.get("id"), "note": note}

    @streamable(
        running_text="Attaching note to document...",
        completed_text="Note attached to the document.",
        params_transform=hide,
        response_transform=hide,
    )
    async def create_document_note(
        self,
        document_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """Attach a note to a specific document.

        Use this to annotate a document with a note. Provide the document's
        id (uri_hash). The note is private to the user.

        Args:
            document_id: The document ID (uri_hash) to attach the note to.
            content: The note text (markdown supported).

        Returns:
            Dict with 'note_id' and the created note.
        """
        note = await self.__notes_service.create_note(
            content=content,
            object_type="document",
            object_id=document_id,
        )
        return {"status": "created", "note_id": note.get("id"), "note": note}

    @streamable(
        running_text="Updating note...",
        completed_text="Note updated.",
        params_transform=hide,
        response_transform=hide,
    )
    async def update_note(self, note_id: int, content: str) -> Dict[str, Any]:
        """Edit the content of an existing note.

        Replaces the note's text while keeping its attachment — a note
        attached to a document stays attached to that document.

        Args:
            note_id: The note ID to edit (from create_note or list_my_notes).
            content: The new note text (markdown supported).

        Returns:
            Dict confirming the update.
        """
        await self.__notes_service.update_note(note_id=note_id, content=content)
        return {"status": "updated", "note_id": note_id}

    @streamable(
        running_text="Deleting note...",
        completed_text="Note deleted.",
        params_transform=hide,
        response_transform=hide,
    )
    async def delete_note(self, note_id: int) -> Dict[str, Any]:
        """Delete a note.

        Args:
            note_id: The note ID to delete (from create_note or list_my_notes).

        Returns:
            Dict confirming the deletion.
        """
        await self.__notes_service.delete_note(note_id)
        return {"status": "deleted", "note_id": note_id}

    @streamable(
        running_text="Listing your notes...",
        completed_text=(
            "Found {{ result_count }} note{{ 's' if result_count != 1 else '' }}."
        ),
        response_transform=hide,
    )
    async def list_my_notes(
        self,
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """List the user's notes with pagination.

        Args:
            page: Page number (default: 1).
            page_size: Number of notes per page.

        Returns:
            Dict with 'results' (list of notes, each including its 'id') and
            'result_count'.
        """
        try:
            notes = await self.__notes_service.retrieve_notes(
                page=page,
                page_size=page_size or self.__config.list_page_size,
                created_by_me=True,
            )
        except Exception as e:
            logger.exception(f"Error listing notes: {e}")
            raise Exception(f"Could not list your notes: {e}") from e
        return {"results": notes, "result_count": len(notes)}

    @streamable(
        running_text="Searching notes...",
        completed_text=(
            "Found {{ results|length }} note{{ 's' if results|length != 1 else '' }}."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def search_notes(
        self,
        query: str,
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search notes by content: the user's own notes and the notes
        colleagues shared with them.

        Colleagues capture how their team or site works as notes: who owns or
        decides something, where something is kept, internal limits, and why
        things are done. `search` does not cover notes, so use this tool for
        such questions too.

        Args:
            query: Natural language search query.
            page: Result page number (default: 1).
            page_size: Number of results per page.

        Returns:
            Dict with 'results' (each with 'note_id', 'content', 'created' and
            'source_url'), 'total_hits' and 'page', plus a 'message' when every
            note on the page was a saved chat conversation and was left out.
            'total_hits' is the index's count, taken before duplicate chunks and
            chat notes are removed, so it can exceed the notes returned.
        """
        hit_fields = {
            "note_id": self.__config.search_id_field,
            "content": self.__config.search_content_field,
            "created": self.__config.search_created_at_field,
        }
        if self.__index_tools is None:
            raise ValueError("search_notes needs an IndexToolsSource.")
        result = await self.__index_tools.search_notes(
            query=query,
            filters=self.__config.search_filters,
            page=page,
            page_size=page_size,
            metadata_fields=list(hit_fields.values()),
        )
        hits = result.get("results", [])
        notes: List[Dict[str, Any]] = []
        seen_ids = set()
        excluded_count = 0
        for hit in hits:
            metadata = hit.get("metadata", {})
            note = {
                key: metadata.get(field.rsplit(".", 1)[-1])
                for key, field in hit_fields.items()
            }
            # Chunk retrieval can return several hits for one note.
            if not note["note_id"] or note["note_id"] in seen_ids:
                continue
            seen_ids.add(note["note_id"])
            is_chat = (note["content"] or "").lstrip().startswith(_CHAT_NOTE_PREFIX)
            if self.__config.exclude_chat_notes and is_chat:
                excluded_count += 1
                continue
            note["source_url"] = hit.get("source_url")
            if hit.get("short_id"):
                note["short_id"] = hit["short_id"]
            notes.append(note)
        total_hits = result.get("total_hits", 0)
        response: Dict[str, Any] = {
            "results": notes,
            "total_hits": total_hits,
            "page": result.get("page", page),
        }
        if not notes and excluded_count:
            message = (
                f"All {excluded_count} notes on this page were saved chat "
                "conversations and were left out"
            )
            if total_hits > page * (page_size or len(hits)):
                message += f". Try page {page + 1} or a larger page_size."
            else:
                message += ", and no other notes matched."
            response["message"] = message
        return response


class NoteToolsFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notes_service: NotesService,
        note_tools_source_configuration: NoteToolsSourceConfiguration = (
            NoteToolsSourceConfiguration()
        ),
        index_tools: Optional[IndexToolsSource] = None,
    ) -> NoteToolsSource:
        return NoteToolsSource(
            notes_service=notes_service,
            note_tools_source_configuration=note_tools_source_configuration,
            index_tools=index_tools,
        )
