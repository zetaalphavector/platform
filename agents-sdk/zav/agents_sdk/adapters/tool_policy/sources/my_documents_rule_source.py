import base64
import hashlib
import json
import mimetypes
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ToolCallRequest,
)
from zav.agents_sdk.adapters.tool_policy.tool_policy_source import ToolPolicySource
from zav.agents_sdk.adapters.user_documents.user_documents_service import (
    UserDocumentCreateRequest,
    UserDocumentMetadata,
    UserDocumentsService,
    create_new_document,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.mcp_content import (
    FileContent,
    file_positions,
    replace_files,
)
from zav.agents_sdk.domain.tool_calls import parse_result_dict
from zav.agents_sdk.domain.tools import ResultDecision, Tool

MY_DOCUMENTS_REFERENCE_PREFIX = "my-documents://"


class MyDocumentsRuleSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Replace `my-documents://<document_id>` in MCP tool arguments with "
            "the file's content, and save the files MCP tools return to My "
            "Documents."
        ),
    )
    server_names: List[str] = Field(
        default_factory=list,
        description=(
            "The MCP servers whose tool calls may receive file content and whose "
            "files are saved to My Documents."
        ),
    )


def _file_name(content: FileContent) -> str:
    # The URI's last segment (a server may send file:///<name>, or
    # file:///download without a file name); an image or audio part has none.
    name = unquote((content.uri or "").rsplit("/", 1)[-1]) or "file"
    if os.path.splitext(name)[1]:
        return name
    extension = mimetypes.guess_extension(content.mime_type or "")
    return f"{name}{extension or ''}"


class MyDocumentsRuleSource(ToolPolicySource):
    """Moves files between My Documents and the configured MCP servers, so the
    model handles references instead of bytes it can neither write nor read:
    a `my-documents://<document_id>` argument is sent as the file's content,
    and a file a tool returns is saved to My Documents and returned as its
    reference."""

    source_name = "my_documents"
    # Saving a returned file is a side effect: run after every rule that may
    # block the result, so a blocked file is never saved.
    specificity = 100
    # A view reads a returned file's bytes itself and cannot read a reference.
    callers = ("direct", "sandbox")
    source_description = (
        "MCP tools receive My Documents files and save the files they return there."
    )

    def __init__(
        self,
        user_documents_service: UserDocumentsService,
        configuration: MyDocumentsRuleSourceConfiguration,
    ):
        self.enabled = configuration.enabled
        self.__user_documents_service = user_documents_service
        self.__server_names = set(configuration.server_names)
        # Files saved by this agent, by name and content: reading the same file
        # again reuses its document instead of saving "<name> (2)".
        self.__saved: Dict[Tuple[str, str], Dict[str, Any]] = {}

    async def prepare_call(
        self, call: ToolCallRequest, tool: Tool
    ) -> Optional[Dict[str, Any]]:
        params = call.function_call_request.params
        if not self.__serves(tool) or not params:
            return None
        return {key: await self.__resolve(value) for key, value in params.items()}

    async def check_result(
        self,
        call: ToolCallRequest,
        tool: Tool,
        result: Any,
        transcript: Optional[List[ChatCompletion]],
    ) -> Optional[ResultDecision]:
        payload = parse_result_dict(result) if self.__serves(tool) else None
        files = file_positions(payload)
        if payload is None or not files:
            return None
        saved = {i: await self.__saved_part(content) for i, content in files.items()}
        return ResultDecision.transform(json.dumps(replace_files(payload, saved)))

    def serves_server(self, server_name: str) -> bool:
        return server_name in self.__server_names

    def __serves(self, tool: Tool) -> bool:
        origin = tool.origin
        return (
            origin is not None
            and origin.provider == "mcp"
            and self.serves_server(origin.source)
        )

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

    async def __saved_part(self, content: FileContent) -> Dict[str, Any]:
        name = _file_name(content)
        key = (name, hashlib.sha256(content.data.encode("ascii")).hexdigest())
        saved = self.__saved.get(key)
        if saved is None:
            saved = await self.__save(name, content)
            if "document_id" in saved:
                self.__saved[key] = saved
        return {"type": "text", "text": json.dumps(saved)}

    async def __save(self, name: str, content: FileContent) -> Dict[str, Any]:
        try:
            document = await create_new_document(
                self.__user_documents_service,
                UserDocumentCreateRequest(
                    document_metadata=UserDocumentMetadata(title=name),
                    content_file_name=name,
                    base64_content=content.data,
                ),
            )
        except Exception as e:
            logger.exception(f"Could not save {name} to My Documents: {e}")
            return {"error": f"The file {name} could not be saved to My Documents: {e}"}
        return {
            "saved_to_my_documents": True,
            "document_id": document.uri_hash,
            "reference": f"{MY_DOCUMENTS_REFERENCE_PREFIX}{document.uri_hash}",
            "name": name,
            "content_type": content.mime_type,
            "note": (
                "Read its content with read_document once it is indexed, which "
                "takes up to a few minutes. Pass the reference as a tool argument "
                "to send the file on."
            ),
        }


class MyDocumentsRuleSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        user_documents_service: UserDocumentsService,
        my_documents_rule_source_configuration: MyDocumentsRuleSourceConfiguration = (
            MyDocumentsRuleSourceConfiguration()
        ),
    ) -> MyDocumentsRuleSource:
        return MyDocumentsRuleSource(
            user_documents_service=user_documents_service,
            configuration=my_documents_rule_source_configuration,
        )
