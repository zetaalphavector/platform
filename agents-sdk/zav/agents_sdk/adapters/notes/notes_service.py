from typing import Dict, List, Literal, Optional

from zav.saved_results import ApiClient, Configuration
from zav.saved_results.apis import NotesApi
from zav.saved_results.model.note_form import NoteForm
from zav.saved_results.model.note_item import NoteItem
from zav.saved_results.model.note_object_type import NoteObjectType
from zav.saved_results.model.paginated_tag_notes import PaginatedTagNotes
from zav.saved_results.model.sharing_policy import SharingPolicy
from zav.saved_results.model.tag_note_form import TagNoteForm
from zav.saved_results.model.tag_note_item import TagNoteItem
from zav.saved_results.model.uuid_string import UUIDString

from zav.agents_sdk.adapters.async_wrapper import asyncify
from zav.agents_sdk.adapters.error_handling import handle_api_errors
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.request_headers import RequestHeaders


class NotesService:
    def __init__(
        self,
        api_client: ApiClient,
        request_headers: RequestHeaders,
        tenant: str,
        index_id: Optional[str] = None,
    ) -> None:
        if request_headers.authorization:
            api_client.set_default_header(
                "Authorization", request_headers.authorization
            )
        if request_headers.x_auth:
            api_client.set_default_header("X-Auth", request_headers.x_auth)

        self.__notes = NotesApi(api_client)
        self.__internal_headers = request_headers.dict(
            exclude_none=True, exclude={"authorization", "x_auth"}
        )
        if (
            request_headers.authorization
            or request_headers.x_auth
            or api_client.default_headers.get("Authorization")
            or api_client.default_headers.get("X-Auth")
        ):
            if "requester_uuid" not in self.__internal_headers:
                self.__internal_headers["requester_uuid"] = UUIDString("")
            if "user_roles" not in self.__internal_headers:
                self.__internal_headers["user_roles"] = ""
        self.__tenant = tenant
        self.__index_id = index_id
        super().__init__()

    @handle_api_errors
    async def create_note(
        self,
        content: str,
        object_type: str = "free",
        object_id: Optional[str] = None,
    ) -> Dict:
        note_form = NoteForm(
            content=content,
            object_type=NoteObjectType(object_type),
            annotation_highlight=None,
            sharing=SharingPolicy(),
            object_id=object_id,
        )
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
        }
        response: NoteItem = await asyncify(self.__notes.create_note)(
            note_form=note_form,
            tenant=self.__tenant,
            **params,
            **self.__internal_headers,
        )
        return response.to_dict()

    @handle_api_errors
    async def retrieve_notes(
        self,
        note_object_type: Optional[str] = None,
        note_object_id: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        created_by_me: Optional[bool] = None,
    ) -> List:
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
            **({"page": page} if page is not None else {}),
            **({"page_size": page_size} if page_size is not None else {}),
            **({"created_by_me": created_by_me} if created_by_me is not None else {}),
            **(
                {"note_object_id": note_object_id} if note_object_id is not None else {}
            ),
            **(
                {"note_object_type": NoteObjectType(note_object_type)}
                if note_object_type is not None
                else {}
            ),
        }
        response = await asyncify(self.__notes.filter_notes)(
            tenant=self.__tenant,
            **params,
            **self.__internal_headers,
        )
        resp = response.to_dict().get("results", [])
        return [item.to_dict() if hasattr(item, "to_dict") else item for item in resp]

    @handle_api_errors
    async def retrieve_note(
        self,
        note_id: int,
    ) -> Dict:
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
        }
        response: NoteItem = await asyncify(self.__notes.filter_notes)(
            note_ids=[note_id],
            tenant=self.__tenant,
            **params,
            **self.__internal_headers,
        )
        notes = response.to_dict().get("results")
        return notes[0] if notes else {}

    @handle_api_errors
    async def update_note(
        self,
        note_id: int,
        content: str,
    ) -> None:
        # replace_note is a full PUT, so preserve the note's existing type and
        # attachment (a document-note keeps its object_id instead of being
        # detached). Sharing follows the owner-private default, as on create.
        # retrieve_note returns object_type already as a NoteObjectType, so pass
        # it through rather than re-wrapping it.
        existing = await self.retrieve_note(note_id)
        existing_object_type = existing.get("object_type")
        object_type = (
            existing_object_type
            if isinstance(existing_object_type, NoteObjectType)
            else NoteObjectType(existing_object_type or "free")
        )
        note_form = NoteForm(
            content=content,
            object_type=object_type,
            annotation_highlight=None,
            sharing=SharingPolicy(),
            object_id=existing.get("object_id"),
        )
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
        }
        await asyncify(self.__notes.replace_note)(
            id=note_id,
            note_form=note_form,
            tenant=self.__tenant,
            **params,
            **self.__internal_headers,
        )

    @handle_api_errors
    async def delete_note(
        self,
        note_id: int,
    ) -> None:
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
        }
        await asyncify(self.__notes.delete_note)(
            id=note_id,
            tenant=self.__tenant,
            **params,
            **self.__internal_headers,
        )

    @handle_api_errors
    async def create_tag_note(
        self,
        tag_type: Literal["own", "shared", "following"],
        tag_id: int,
        content: str,
    ) -> Dict:
        tag_note_form = TagNoteForm(
            content=content,
            sharing=SharingPolicy(),
        )
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
        }
        response: TagNoteItem = await asyncify(self.__notes.create_tag_note)(
            tag_type=tag_type,
            id=tag_id,
            tag_note_form=tag_note_form,
            tenant=self.__tenant,
            **params,
            **self.__internal_headers,
        )
        return response.to_dict()

    @handle_api_errors
    async def retrieve_tag_notes(
        self,
        tag_type: Literal["own", "shared", "following"],
        tag_id: int,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> List:
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
            **({"page": page} if page is not None else {}),
            **({"page_size": page_size} if page_size is not None else {}),
        }
        result: PaginatedTagNotes = await asyncify(self.__notes.tag_notes_filter)(
            tag_type=tag_type,
            id=tag_id,
            tenant=self.__tenant,
            **params,
            **self.__internal_headers,
        )
        resp = result.to_dict().get("results", [])
        return [item.to_dict() if hasattr(item, "to_dict") else item for item in resp]

    @handle_api_errors
    async def retrieve_favorite_tag_notes(
        self,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> List:
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
            **({"page": page} if page is not None else {}),
            **({"page_size": page_size} if page_size is not None else {}),
        }
        result: PaginatedTagNotes = await asyncify(
            self.__notes.favorite_tag_notes_filter
        )(
            tenant=self.__tenant,
            **params,
            **self.__internal_headers,
        )
        resp = result.to_dict().get("results", [])
        return [item.to_dict() if hasattr(item, "to_dict") else item for item in resp]

    @handle_api_errors
    async def create_favorite_tag_note(
        self,
        content: str,
    ) -> Dict:
        tag_note_form = TagNoteForm(
            content=content,
            sharing=SharingPolicy(),
        )
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
        }
        response: TagNoteItem = await asyncify(self.__notes.create_favorite_tag_note)(
            tag_note_form=tag_note_form,
            tenant=self.__tenant,
            **params,
            **self.__internal_headers,
        )
        return response.to_dict()


def _api_config(host: str, retries: Optional[int] = None) -> Configuration:
    config = Configuration(host=host)
    config.retries = retries  # type: ignore
    return config


class NotesServiceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        request_headers: RequestHeaders,
        zav_notes_api_host: str = "https://api.zeta-alpha.com/v0/service",
        tenant: str = "zetaalpha",
        index_id: Optional[str] = None,
        authorization: Optional[str] = None,
        x_auth: Optional[str] = None,
        retries: Optional[int] = None,
    ) -> NotesService:
        configuration = _api_config(zav_notes_api_host, retries)
        api_client = ApiClient(configuration)
        if authorization:
            api_client.set_default_header("Authorization", authorization)
        if x_auth:
            api_client.set_default_header("X-Auth", x_auth)
        return NotesService(
            api_client,
            request_headers,
            tenant,
            index_id,
        )
