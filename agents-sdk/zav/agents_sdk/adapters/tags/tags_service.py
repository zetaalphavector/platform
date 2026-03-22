from typing import Dict, List, Literal, Optional

from zav.saved_results import ApiClient, Configuration
from zav.saved_results.apis import TagsApi
from zav.saved_results.model.guid_string import GUIDString
from zav.saved_results.model.paginated_tagged_documents import PaginatedTaggedDocuments
from zav.saved_results.model.paginated_tags import PaginatedTags
from zav.saved_results.model.sharing_policy import SharingPolicy
from zav.saved_results.model.tag_form import TagForm
from zav.saved_results.model.tag_item_settings import TagItemSettings
from zav.saved_results.model.tag_name import TagName
from zav.saved_results.model.tagged_document_form import TaggedDocumentForm
from zav.saved_results.model.uuid_string import UUIDString

from zav.agents_sdk.adapters.async_wrapper import asyncify
from zav.agents_sdk.adapters.error_handling import handle_api_errors
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.request_headers import RequestHeaders


class TagsService:
    def __init__(
        self,
        api_client: ApiClient,
        request_headers: RequestHeaders,
        tenant: str,
        max_documents_per_page: int,
        index_id: Optional[str] = None,
        tags_service_page_size: int = 100,
    ) -> None:
        if request_headers.authorization:
            api_client.set_default_header(
                "Authorization", request_headers.authorization
            )
        if request_headers.x_auth:
            api_client.set_default_header("X-Auth", request_headers.x_auth)

        self.__tags = TagsApi(api_client)
        self.__internal_headers = request_headers.dict(
            exclude_none=True, exclude={"authorization", "x_auth"}
        )
        if (
            request_headers.authorization
            or api_client.default_headers.get("Authorization")
            or request_headers.x_auth
            or api_client.default_headers.get("X-Auth")
        ):
            # This is the case when the external API is being called
            if "requester_uuid" not in self.__internal_headers:
                self.__internal_headers["requester_uuid"] = UUIDString("")
            if "user_roles" not in self.__internal_headers:
                self.__internal_headers["user_roles"] = ""
        self.__tenant = tenant
        self.__index_id = index_id
        self.__tags_service_page_size = int(tags_service_page_size)
        self.__request_params = {
            **({"index_id": self.__index_id} if self.__index_id else {}),
            **self.__internal_headers,
        }
        self.__max_documents_per_page = int(max_documents_per_page)
        super().__init__()

    @handle_api_errors
    async def get_tags(self, page_size: Optional[int] = None) -> List[Dict]:
        results: List[Dict] = []
        page: int = 1
        page_size = (
            int(page_size) if page_size is not None else self.__tags_service_page_size
        )

        while True:
            tag_response: PaginatedTags = await asyncify(self.__tags.tags_filter_all)(
                tenant=self.__tenant,
                page=page,
                page_size=page_size,
                **self.__request_params,
            )
            tag_results = tag_response.to_dict()
            page_results = [r.to_dict() for r in tag_results.get("results", [])]
            results.extend(page_results)
            count = tag_results.get("count", 0)
            if not page_results or len(results) >= count:
                break
            page += 1

        return results

    @handle_api_errors
    async def get_tag_by_id(self, tag_id: int) -> Dict | None:
        """Get a specific tag by its ID.

        Args:
            tag_id: The ID of the tag to retrieve.

        Returns:
            Optional[Dict]: Tag metadata including name, description, document_count,
            etc. If the tag is not found, returns None.
        """
        tag_response = await asyncify(self.__tags.tags_filter_all)(
            tenant=self.__tenant,
            tag_ids=[tag_id],
            **self.__request_params,
        )
        tag_results = tag_response.to_dict().get("results", [])
        if not tag_results:
            return None
        return tag_results[0].to_dict()

    async def get_tag_type_by_id(
        self, tag_id: int
    ) -> Literal["own", "shared", "following"] | None:
        tag = await self.get_tag_by_id(tag_id)
        if tag is None:
            return None
        tag_type = tag["tag_type"].lower()
        if tag_type not in ("own", "shared", "following"):
            return None
        return tag_type

    @handle_api_errors
    async def get_tagged_documents_paginated(
        self, tag_id: int, tag_type: str, page: int = 1, page_size: Optional[int] = None
    ) -> Dict:
        """Get tagged documents with pagination.

        Args:
            tag_id: The ID of the tag
            tag_type: Type of tag (own, shared, following, favourites)
            page: Page number (1-indexed)
            page_size: Number of results per page. If None, uses default from config.

        Returns:
            Dict with 'results' (list of documents) and pagination info
        """
        params = self.__request_params.copy()
        if (tag_type := tag_type.lower()) == "favourites":
            api = self.__tags.favorite_tag_documents_filter
        else:
            api = self.__tags.tagged_documents_filter
            params["id"] = tag_id
            params["tag_type"] = tag_type

        tag_response: PaginatedTaggedDocuments = await asyncify(api)(
            tenant=self.__tenant,
            page=page,
            page_size=page_size or self.__max_documents_per_page,
            **params,
        )
        tag_results = tag_response.to_dict()
        return {
            "results": [r.to_dict() for r in tag_results.get("results", [])],
            "total_count": tag_results.get("count", 0),
            "next": tag_results.get("next"),
            "previous": tag_results.get("previous"),
        }

    @handle_api_errors
    async def get_tagged_documents(self, tag_id: int, tag_type: str) -> List[Dict]:
        results: List[Dict] = []
        page: Optional[int] = 1
        while page:
            paginated_tag_response = await self.get_tagged_documents_paginated(
                tag_id=tag_id, tag_type=tag_type, page=page
            )
            next = paginated_tag_response.get("next", {})
            page = next.get("page") if next else None
            results.extend(paginated_tag_response.get("results", []))

        return results

    @handle_api_errors
    async def tag_document(
        self,
        tag_id: int,
        tag_type: Literal["own", "shared", "following"],
        uri_hash: str,
        document_type: Literal["document"] = "document",
        user_order: Optional[int] = None,
    ) -> Dict:
        params = self.__request_params.copy()
        params["id"] = tag_id
        params["tag_type"] = tag_type

        tagged_document_form = TaggedDocumentForm(
            document_id=GUIDString(uri_hash),
            document_type=document_type,
            user_order=user_order,
        )
        params["tagged_document_form"] = tagged_document_form

        tagged_document_response = await asyncify(self.__tags.create_tagged_document)(
            tenant=self.__tenant,
            **params,
        )

        return tagged_document_response.to_dict()

    @handle_api_errors
    async def create_tag(
        self,
        name: str,
        description: Optional[str] = None,
        color: Optional[str] = None,
        sharing: Optional[SharingPolicy] = None,
        settings: Optional[TagItemSettings] = None,
    ) -> Dict:
        """Create a new tag."""
        tag_form = TagForm(
            name=TagName(name),
            description=description,
            color=color,
            settings=(
                settings
                if settings is not None
                else TagItemSettings(
                    newsletter_schedule=None,
                    recommendations_enabled=False,
                )
            ),
            sharing=sharing if sharing is not None else SharingPolicy(),
            user_order=None,
        )

        tag_response = await asyncify(self.__tags.tag_create)(
            tenant=self.__tenant,
            tag_form=tag_form,
            **self.__request_params,
        )

        return tag_response.to_dict()


def _api_config(host: str, retries: Optional[int] = None) -> Configuration:
    config = Configuration(host=host)
    # `None` value in retries means that the default value of `urllib3`
    # will be used, which is 3 retries.
    config.retries = retries  # type: ignore
    return config


class TagsServiceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        request_headers: RequestHeaders,
        zav_saved_results_url: str = "https://api.zeta-alpha.com/v0/service",
        tenant: str = "zetaalpha",
        index_id: Optional[str] = None,
        authorization: Optional[str] = None,
        x_auth: Optional[str] = None,
        retries: Optional[int] = None,
        tags_service_page_size: int = 100,
        max_documents_per_page: int = 20,
    ) -> TagsService:
        config = _api_config(zav_saved_results_url, retries)
        api_client = ApiClient(config)
        if authorization:
            api_client.set_default_header("Authorization", authorization)
        if x_auth:
            api_client.set_default_header("X-Auth", x_auth)
        return TagsService(
            api_client=api_client,
            request_headers=request_headers,
            tenant=tenant,
            index_id=index_id,
            tags_service_page_size=tags_service_page_size,
            max_documents_per_page=max_documents_per_page,
        )
