from typing import Optional

from pydantic import BaseModel
from zav.authentication_service.api.authentication_api import AuthenticationApi
from zav.authentication_service.api.settings_api import SettingsApi
from zav.authentication_service.api_client import ApiClient
from zav.authentication_service.configuration import Configuration
from zav.authentication_service.models import UserSettings, VerifyForm

from zav.agents_sdk.adapters.async_wrapper import asyncify
from zav.agents_sdk.adapters.error_handling import handle_api_errors
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.request_headers import RequestHeaders


class UserInfo(BaseModel):
    uuid: str
    first_name: str
    last_name: str
    email: str
    report_email: bool


class RegistrationService:
    def __init__(
        self,
        api_client: ApiClient,
        request_headers: RequestHeaders,
        tenant: str,
    ) -> None:
        if request_headers.authorization:
            api_client.set_default_header(
                "Authorization", request_headers.authorization
            )
        if request_headers.x_auth:
            api_client.set_default_header("X-Auth", request_headers.x_auth)
        self.__request_headers = request_headers
        self.__users_settings = SettingsApi(api_client)
        self.__authentication = AuthenticationApi(api_client)
        self.__internal_headers = request_headers.dict(
            exclude_none=True,
            exclude={"authorization", "x_auth", "user_roles", "requester_uuid"},
        )
        self.__user_uuid = request_headers.requester_uuid
        self.__tenant = tenant

    @handle_api_errors
    async def __get_user_uuid(self) -> str:
        verify_form: Optional[VerifyForm] = None
        if (bearer := self.__request_headers.authorization) is not None:
            # Remove Bearer from token
            verify_form = VerifyForm(jwt=bearer[7:])
        elif (x_auth := self.__request_headers.x_auth) is not None:
            verify_form = VerifyForm(jwt=x_auth)
        if verify_form is None:
            raise ValueError("No authorization token found in request headers")
        user_payload = await asyncify(self.__authentication.verify)(
            verify_form=verify_form
        )
        return user_payload.sub

    @handle_api_errors
    async def get_user_info(self) -> UserInfo:
        """Get users info.

        Args:
            user_uuid: user UUID

        Returns:
            UserInfo.

        Raises:
            NotFound: User not found
            BadRequest: Bad request
            UnknownException: Unknown exception
        """
        if self.__user_uuid is None:
            # This is the case when the external API is being called
            self.__user_uuid = await self.__get_user_uuid()
        user_info: UserSettings = await asyncify(
            self.__users_settings.settings_retrieve
        )(
            requester_uuid=self.__user_uuid,
            tenant=self.__tenant,
            **self.__internal_headers,
        )
        user_email = (
            str(user_info.email).replace("keycloak/", "")
            if str(user_info.email).startswith("keycloak/")
            else str(user_info.email)
        )
        return UserInfo(
            uuid=self.__user_uuid,
            first_name=user_info.first_name,
            last_name=user_info.last_name,
            email=user_email,
            report_email=user_info.report_email,
        )

    @handle_api_errors
    async def get_user_info_by_uuid(self, user_uuid: str) -> UserInfo:
        """Get user info by UUID.

        Args:
            user_uuid: user UUID

        Returns:
            UserInfo.

        Raises:
            NotFound: User not found
            BadRequest: Bad request
            UnknownException: Unknown exception
        """
        user_info: UserSettings = await asyncify(
            self.__users_settings.settings_retrieve
        )(
            requester_uuid=user_uuid,
            tenant=self.__tenant,
            **self.__internal_headers,
        )
        user_email = (
            str(user_info.email).replace("keycloak/", "")
            if str(user_info.email).startswith("keycloak/")
            else str(user_info.email)
        )
        return UserInfo(
            uuid=user_uuid,
            first_name=user_info.first_name,
            last_name=user_info.last_name,
            email=user_email,
            report_email=user_info.report_email,
        )


class RegistrationServiceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        request_headers: RequestHeaders,
        zav_registration_api_host: str = "https://api.zeta-alpha.com/v0/service",
        tenant: str = "zetaalpha",
        authorization: Optional[str] = None,
        x_auth: Optional[str] = None,
    ) -> RegistrationService:
        configuration = Configuration(host=zav_registration_api_host)
        api_client = ApiClient(configuration)
        if authorization:
            api_client.set_default_header("Authorization", authorization)
        if x_auth:
            api_client.set_default_header("X-Auth", x_auth)
        return RegistrationService(api_client, request_headers, tenant)
