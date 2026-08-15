import enum
from typing import Any, Dict, List, Optional, Union

from zav.prompt_completion import ChatMessage as PcChatMessage
from zav.prompt_completion import ChatMessageSender
from zav.prompt_completion import ChatResponse as PcChatResponse
from zav.prompt_completion import FunctionCallRequest as PcFunctionCallRequest
from zav.prompt_completion import FunctionCallResponse as PcFunctionCallResponse
from zav.prompt_completion import ToolCallRequest as PcToolCallRequest
from zav.prompt_completion import ToolCallResponse as PcToolCallResponse
from zav.pydantic_compat import PYDANTIC_V2, BaseModel, ConfigDict, Field

from zav.agents_sdk.domain.chat_message import (
    ChatMessage,
)
from zav.agents_sdk.domain.chat_message import (
    ChatMessageSender as AgentChatMessageSender,
)
from zav.agents_sdk.domain.chat_message import (
    ContentPart,
    FunctionCallRequest,
)


class ToolCallRequest(BaseModel):
    id: str
    function_call_request: FunctionCallRequest


class FunctionCallResponse(BaseModel):
    name: str
    response: Optional[str] = None


class ToolCallResponse(BaseModel):
    id: str
    response: Optional[str] = None


class ChatCompletionSender(str, enum.Enum):
    USER = "user"
    BOT = "bot"
    FUNCTION = "function"
    TOOL = "tool"
    DEVELOPER = "developer"
    SYSTEM = "system"


class ChatCompletion(BaseModel):
    sender: ChatCompletionSender
    content: str
    image_uri: Optional[str] = None
    function_call_request: Optional[FunctionCallRequest] = None
    function_call_response: Optional[FunctionCallResponse] = None
    tool_call_requests: Optional[List[ToolCallRequest]] = None
    tool_call_responses: Optional[List[ToolCallResponse]] = None
    response_replay_items: Optional[List[Dict[str, Any]]] = None
    reasoning_summary: Optional[str] = None

    @classmethod
    def from_chat_message(cls, chat_message: ChatMessage) -> "ChatCompletion":
        parts_text = (
            "".join([part.text for part in chat_message.content_parts if part.text])
            if chat_message.content_parts
            else ""
        )
        content = parts_text or chat_message.content or ""
        return cls(
            sender=ChatCompletionSender(chat_message.sender),
            content=content,
            image_uri=chat_message.image_uri,
            function_call_request=chat_message.function_call_request,
            function_call_response=(
                FunctionCallResponse(
                    name=chat_message.function_call_response.name,
                    response=chat_message.function_call_response.result,
                )
                if chat_message.function_call_response
                else None
            ),
            reasoning_summary=chat_message.reasoning_summary,
        )

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True


class ChatResponse(BaseModel):
    error: Optional[Exception] = None
    chat_completion: Optional[ChatCompletion] = None
    content_parts: List[ContentPart] = Field(default_factory=list)

    if PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:

        class Config:
            arbitrary_types_allowed = True

    def to_chat_message(self) -> Optional[ChatMessage]:
        if self.chat_completion is None:
            if not self.content_parts:
                return None
            return ChatMessage(
                sender=AgentChatMessageSender.BOT,
                content="",
                content_parts=self.content_parts,
            )

        if self.chat_completion.sender == ChatCompletionSender.TOOL:
            if not self.content_parts:
                return None
            return ChatMessage(
                sender=AgentChatMessageSender.BOT,
                content=self.chat_completion.content,
                content_parts=self.content_parts,
            )

        message = ChatMessage.parse_obj(self.chat_completion)
        if self.content_parts:
            message.content_parts = self.content_parts
        return message


def parse_chat_message(message: Union[ChatMessage, ChatCompletion]) -> PcChatMessage:
    parts_text = ""
    if isinstance(message, ChatMessage) and message.content_parts:
        parts_text = "".join([part.text for part in message.content_parts if part.text])
    content = parts_text or message.content

    return PcChatMessage(
        sender=ChatMessageSender(message.sender),
        content=content,
        image_uri=message.image_uri,
        function_call_request=(
            PcFunctionCallRequest(
                function_name=message.function_call_request.name,
                function_params=(message.function_call_request.params),
            )
            if message.function_call_request
            else None
        ),
        function_call_response=(
            PcFunctionCallResponse(
                function_name=message.function_call_response.name,
                function_response=(message.function_call_response.response),
            )
            if isinstance(message, ChatCompletion) and message.function_call_response
            else None
        ),
        tool_call_requests=(
            [
                PcToolCallRequest(
                    id=tcr.id,
                    function_call_request=PcFunctionCallRequest(
                        function_name=tcr.function_call_request.name,
                        function_params=tcr.function_call_request.params,
                    ),
                )
                for tcr in message.tool_call_requests
            ]
            if isinstance(message, ChatCompletion) and message.tool_call_requests
            else None
        ),
        tool_call_responses=(
            [
                PcToolCallResponse(
                    id=tool_call_response.id,
                    tool_response=tool_call_response.response,
                )
                for tool_call_response in message.tool_call_responses
            ]
            if isinstance(message, ChatCompletion) and message.tool_call_responses
            else None
        ),
        response_replay_items=(
            message.response_replay_items
            if isinstance(message, ChatCompletion)
            else None
        ),
        reasoning_summary=message.reasoning_summary,
    )


def convert_chat_response(chat_response: PcChatResponse) -> ChatResponse:
    chat_response_message = chat_response.chat_message
    chat_completion = (
        ChatCompletion(
            sender=ChatCompletionSender(chat_response_message.sender),
            content=chat_response_message.content,
            image_uri=chat_response_message.image_uri,
            function_call_request=(
                FunctionCallRequest(
                    name=function_call_request.function_name,
                    params=(function_call_request.function_params),
                )
                if (
                    function_call_request := chat_response_message.function_call_request
                )
                else None
            ),
            function_call_response=(
                FunctionCallResponse(
                    name=function_call_response.function_name,
                    response=function_call_response.function_response,
                )
                if (
                    function_call_response := (
                        chat_response_message.function_call_response
                    )
                )
                else None
            ),
            tool_call_requests=(
                [
                    ToolCallRequest(
                        id=tcr.id,
                        function_call_request=FunctionCallRequest(
                            name=tcr.function_call_request.function_name,
                            params=tcr.function_call_request.function_params,
                        ),
                    )
                    for tcr in tool_call_requests
                ]
                if (tool_call_requests := chat_response_message.tool_call_requests)
                else None
            ),
            tool_call_responses=(
                [
                    ToolCallResponse(
                        id=tool_call_response.id,
                        response=tool_call_response.tool_response,
                    )
                    for tool_call_response in tool_call_responses
                ]
                if (tool_call_responses := chat_response_message.tool_call_responses)
                else None
            ),
            response_replay_items=chat_response_message.response_replay_items,
            reasoning_summary=chat_response_message.reasoning_summary,
        )
        if chat_response_message
        else None
    )
    return ChatResponse(
        error=chat_response.error,
        chat_completion=chat_completion,
    )
