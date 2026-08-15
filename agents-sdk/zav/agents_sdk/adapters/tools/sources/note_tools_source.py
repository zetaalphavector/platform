from typing import Any, Dict, List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.notes.notes_service import NotesService
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable


class NoteToolsSourceConfiguration(BaseModel):
    enabled: bool = False
    list_page_size: int = 20


class NoteToolsSource(ToolsSource):

    source_name = "note_tools"

    def __init__(
        self,
        notes_service: NotesService,
        note_tools_source_configuration: NoteToolsSourceConfiguration,
    ):
        self.__notes_service = notes_service
        self.__config = note_tools_source_configuration
        self.enabled = note_tools_source_configuration.enabled

    async def get_tools(self) -> List[Tool]:
        return [
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


class NoteToolsFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notes_service: NotesService,
        note_tools_source_configuration: NoteToolsSourceConfiguration = (
            NoteToolsSourceConfiguration()
        ),
    ) -> NoteToolsSource:
        return NoteToolsSource(
            notes_service=notes_service,
            note_tools_source_configuration=note_tools_source_configuration,
        )
