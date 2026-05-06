import json
import mimetypes
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from zav.common import mimetypes as _  # noqa: F401 — registers model/jt MIME fix
from zav.logging import logger
from zav.pydantic_compat import BaseModel, ConfigDict, Field
from zav.user_documents_connector import ApiClient, Configuration
from zav.user_documents_connector.apis import ExternalDocumentsApi, UserDocumentsApi
from zav.user_documents_connector.models import (
    ItemAccessRights,
    UserDocumentForm,
    UserDocumentItem,
)
from zav.user_documents_connector.models import (
    UserDocumentMetadata as ZavUserDocumentMetadata,
)

from zav.agents_sdk.adapters.async_wrapper import asyncify
from zav.agents_sdk.adapters.error_handling import handle_api_errors
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.request_headers import RequestHeaders


class UserDocumentAccessRight(str, Enum):
    own = "own"
    org = "org"


class UserDocumentStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class UserDocumentResource(BaseModel):
    resource_type: str
    resource_value: str


class UserDocumentMetadata(BaseModel):
    model_config = ConfigDict(
        extra="allow", json_schema_extra={"additionalProperties": True}
    )

    title: Optional[str] = None
    year: Optional[int] = None
    authors: Optional[List[str]] = None
    description: Optional[str] = None
    source: Optional[str] = None
    resources: Optional[List[UserDocumentResource]] = None
    allow_ai_extraction: Optional[bool] = None
    subject: Optional[List[str]] = None


class OrgSharingPolicy(BaseModel):
    org_id: str
    permission: str


class UserSharingPolicy(BaseModel):
    user_uuid: str
    permission: str


class UserDocumentSharing(BaseModel):
    orgs: List[OrgSharingPolicy] = Field(default_factory=list)
    users: List[UserSharingPolicy] = Field(default_factory=list)


class UserDocumentCreateRequest(BaseModel):
    access_roles: List[str] = [UserDocumentAccessRight.own.value]
    document_metadata: UserDocumentMetadata
    content_file_name: Optional[str] = None
    base64_content: Optional[str] = None
    sharing: Optional[UserDocumentSharing] = UserDocumentSharing()


class UserDocument(BaseModel):
    id: str
    crawled_document_id: str
    owner: str
    access_roles: List[str]
    metadata: UserDocumentMetadata = Field(alias="document_metadata")
    request_created_at: datetime
    request_last_updated_at: datetime
    uri_hash: str
    index_id: str
    tenant: str
    document_type: str
    uri: Optional[str] = None
    content_url: Optional[str] = None
    content_type: Optional[str] = None
    content_file_name: Optional[str] = None
    content_file_size: Optional[int] = None
    status: Optional[UserDocumentStatus] = None
    error_message: Optional[str] = None
    sharing: Optional[UserDocumentSharing] = None


class UserDocumentListResponse(BaseModel):
    documents: List[UserDocument]
    count: int
    page: int
    page_size: int


def _parse_user_document_item(
    user_document: UserDocumentItem,
) -> UserDocument:
    user_document_dict = user_document.to_dict()
    if content_file_name := user_document_dict.get("content_file_name"):
        content_type = (
            mimetypes.guess_type(content_file_name)[0]
        ) or "application/octet-stream"
        del user_document_dict["content_file_name"]
        user_document_dict["content_type"] = content_type
    if uri := user_document_dict.get("uri"):
        if not uri.startswith("https://") and not uri.startswith("http://"):
            user_document_dict["uri"] = None
    return UserDocument(**user_document_dict)


class UserDocumentsService:
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

        self.__user_documents = UserDocumentsApi(api_client)
        self.__external_documents = ExternalDocumentsApi(api_client)
        self.__request_headers = request_headers
        self.__tenant = tenant
        self.__index_id = index_id

    def __user_tenants(self, tenant: str) -> str:
        return json.dumps(dict(tenants=[tenant]))

    def __user_roles_from_request_headers(self) -> str:
        if self.__request_headers.user_roles:
            try:
                user_roles_data = json.loads(self.__request_headers.user_roles)
                roles = user_roles_data.get("roles", [])
            except (json.JSONDecodeError, AttributeError):
                roles = []
        else:
            roles = []

        return json.dumps(dict(roles=roles, role_data=[]))

    @handle_api_errors
    async def create_document(
        self,
        document: UserDocumentCreateRequest,
    ) -> UserDocument:
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
        }
        document_dict = document.dict(exclude_none=True)
        document_form = UserDocumentForm(
            access_roles=[
                ItemAccessRights(access_right)
                for access_right in document_dict["access_roles"]
            ],
            document_metadata=ZavUserDocumentMetadata(
                **document_dict["document_metadata"]
            ),
            base64_content=document_dict["base64_content"],
            content_file_name=document_dict["content_file_name"],
        )
        user_tenants = self.__user_tenants(self.__tenant)
        user_roles = self.__user_roles_from_request_headers()
        requester_uuid = self.__request_headers.requester_uuid or ""
        logger.info(
            f"user_tenants: {user_tenants}, user_roles: {user_roles}, "
            f"requester_uuid: {requester_uuid}"
        )
        user_document = await asyncify(self.__user_documents.create_user_document)(
            tenant=self.__tenant,
            user_document_form=document_form,
            requester_uuid=requester_uuid,
            user_tenants=user_tenants,
            user_roles=user_roles,
            **params,
        )
        return _parse_user_document_item(user_document)

    @handle_api_errors
    async def create_external_document(
        self,
        uri: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        description: Optional[str] = None,
        year: Optional[int] = None,
        date: Optional[str] = None,
        source: Optional[str] = None,
        search_engine: Optional[str] = None,
    ) -> UserDocument:
        metadata_kwargs: Dict[str, Any] = {}
        if title is not None:
            metadata_kwargs["title"] = title
        if authors is not None:
            metadata_kwargs["authors"] = authors
        if description is not None:
            metadata_kwargs["description"] = description
        if year is not None:
            metadata_kwargs["year"] = year
        if source is not None:
            metadata_kwargs["source"] = source
        if search_engine is not None:
            metadata_kwargs["search_engine"] = search_engine
        if date is not None:
            metadata_kwargs["date"] = datetime.fromisoformat(date)

        document_form = UserDocumentForm(
            access_roles=[ItemAccessRights("own")],
            document_metadata=ZavUserDocumentMetadata(**metadata_kwargs),
            uri=uri,
        )

        user_tenants = self.__user_tenants(self.__tenant)
        user_roles = self.__user_roles_from_request_headers()
        requester_uuid = self.__request_headers.requester_uuid or ""
        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
        }

        user_document = await asyncify(
            self.__external_documents.create_external_document
        )(
            tenant=self.__tenant,
            user_document_form=document_form,
            requester_uuid=requester_uuid,
            user_tenants=user_tenants,
            user_roles=user_roles,
            **params,
        )
        return _parse_user_document_item(user_document)

    @handle_api_errors
    async def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> UserDocumentListResponse:
        user_tenants = self.__user_tenants(self.__tenant)
        user_roles = self.__user_roles_from_request_headers()
        requester_uuid = self.__request_headers.requester_uuid or ""

        params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
        }

        response = await asyncify(self.__user_documents.filter_user_documents)(
            tenant=self.__tenant,
            user_tenants=user_tenants,
            requester_uuid=requester_uuid,
            user_roles=user_roles,
            page=page,
            page_size=page_size,
            include_document_status=True,
            **params,
        )

        response_dict = response.to_dict()
        documents = list(map(_parse_user_document_item, response.results))

        return UserDocumentListResponse(
            documents=documents,
            count=response_dict.get("count", 0),
            page=page,
            page_size=page_size,
        )


def _api_config(host: str, retries: Optional[int] = None) -> Configuration:
    config = Configuration(host=host)
    config.retries = retries  # type: ignore
    return config


class UserDocumentsServiceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        request_headers: RequestHeaders,
        zav_user_documents_api_host: str = "https://api.zeta-alpha.com/v0/service",
        tenant: str = "zetaalpha",
        index_id: Optional[str] = None,
        authorization: Optional[str] = None,
        x_auth: Optional[str] = None,
        retries: Optional[int] = None,
    ) -> UserDocumentsService:
        configuration = _api_config(zav_user_documents_api_host, retries)
        api_client = ApiClient(configuration)
        if authorization:
            api_client.set_default_header("Authorization", authorization)
        if x_auth:
            api_client.set_default_header("X-Auth", x_auth)
        return UserDocumentsService(
            api_client,
            request_headers,
            tenant,
            index_id,
        )
