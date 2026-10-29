import base64
from typing import Any, Dict, List

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.mcp.tools_provider import MCPArgumentResolver
from zav.agents_sdk.adapters.user_documents.user_documents_service import (
    UserDocumentsService,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

MY_DOCUMENTS_REFERENCE_PREFIX = "my-documents://"


class MyDocumentsArgumentResolverConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Replace `my-documents://<document_id>` in MCP tool arguments with "
            "the file's content."
        ),
    )
    server_names: List[str] = Field(
        default_factory=list,
        description="The MCP servers whose tool calls may receive file content.",
    )


class MyDocumentsArgumentResolver:
    """Replaces `my-documents://<document_id>` in the arguments of the
    configured servers' tools with the file's bytes, base64-encoded: a model
    cannot pass a file's bytes itself."""

    def __init__(
        self,
        user_documents_service: UserDocumentsService,
        configuration: MyDocumentsArgumentResolverConfiguration,
    ):
        self.__user_documents_service = user_documents_service
        self.__config = configuration

    async def resolve(
        self, server_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.__config.enabled or server_name not in self.__config.server_names:
            return arguments
        return {key: await self.__resolve(value) for key, value in arguments.items()}

    async def __resolve(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith(MY_DOCUMENTS_REFERENCE_PREFIX):
            content = await self.__user_documents_service.retrieve_document_content(
                value[len(MY_DOCUMENTS_REFERENCE_PREFIX) :]
            )
            return base64.b64encode(content).decode("ascii")
        if isinstance(value, dict):
            return {key: await self.__resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [await self.__resolve(item) for item in value]
        return value


class MyDocumentsArgumentResolverFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        user_documents_service: UserDocumentsService,
        my_documents_argument_resolver_configuration: (
            MyDocumentsArgumentResolverConfiguration
        ) = MyDocumentsArgumentResolverConfiguration(),
    ) -> MCPArgumentResolver:
        return MyDocumentsArgumentResolver(
            user_documents_service=user_documents_service,
            configuration=my_documents_argument_resolver_configuration,
        )
