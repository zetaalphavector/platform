import enum
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, overload

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore

from typing_extensions import AsyncIterator, Literal, NotRequired, TypedDict
from zav.llm_domain import LLMModelConfiguration
from zav.llm_tracing import Span

TokenScore = Dict[str, float]


class PromptAnswer(TypedDict):
    text: str


class PromptAnswerWithLogits(PromptAnswer):
    token_scores: List[TokenScore]


class PromptResponse(BaseModel):
    error: Optional[Exception]
    prompt_answer: Optional[Union[PromptAnswer, PromptAnswerWithLogits]]

    class Config:
        arbitrary_types_allowed = True


class ChatMessageSender(str, enum.Enum):
    USER = "user"
    BOT = "bot"
    FUNCTION = "function"
    TOOL = "tool"


class FunctionCallRequest(BaseModel):
    function_name: str
    function_params: Optional[Dict[str, Any]] = None


class FunctionCallResponse(BaseModel):
    function_name: str
    function_response: Optional[str] = None


class ToolCallRequest(BaseModel):
    id: str
    function_call_request: FunctionCallRequest


class ToolCallResponse(BaseModel):
    id: str
    tool_response: Optional[str] = None


class ChatMessage(BaseModel):
    sender: ChatMessageSender
    content: str
    image_uri: Optional[str] = None
    function_call_request: Optional[FunctionCallRequest] = None
    function_call_response: Optional[FunctionCallResponse] = None
    tool_call_requests: Optional[List[ToolCallRequest]] = None
    tool_call_responses: Optional[List[ToolCallResponse]] = None

    def __str__(self) -> str:
        # TODO: This gets logged by the chat service. We should probably obfuscate the
        # parts that contain sensitive information. (eg: self.content is raw user input)
        str_repr = f"[{self.sender.value.upper()}]"
        if self.content:
            str_repr += f" {self.content}"
        if self.image_uri:
            str_repr += f" [img:{self.image_uri[:10]}...]"
        if self.function_call_request:
            fn_name = self.function_call_request.function_name
            fn_params = self.function_call_request.function_params
            str_repr += f" {fn_name}({fn_params})"
        if self.function_call_response:
            fn_name = self.function_call_response.function_name
            fn_response = self.function_call_response.function_response
            str_repr += f" {fn_name}() → {fn_response}"
        if self.tool_call_requests:
            tool_str_reprs = []
            for tool_call_request in self.tool_call_requests:
                fn_name = tool_call_request.function_call_request.function_name
                fn_params = tool_call_request.function_call_request.function_params
                tool_str_reprs.append(f"{fn_name}({fn_params})")
            str_repr += f" [{', '.join(tool_str_reprs)}]"
        if self.tool_call_responses and self.tool_call_requests:
            tool_str_reprs = []
            for tool_call_request, tool_call_response in zip(
                self.tool_call_requests, self.tool_call_responses
            ):
                fn_name = tool_call_request.function_call_request.function_name
                fn_response = tool_call_response.tool_response
                tool_str_reprs.append(f"{fn_name}() → {fn_response}")
            str_repr += f" [{', '.join(tool_str_reprs)}]"

        return str_repr


class ChatResponse(BaseModel):
    error: Optional[Exception]
    chat_message: Optional[ChatMessage]

    class Config:
        arbitrary_types_allowed = True


class BotConversation(BaseModel):
    bot_setup_description: Optional[str]
    messages: List[ChatMessage]


class PromptTooLargeError(Exception):
    """Raised when prompt is too large for the API provider to handle."""

    def __init__(self, message: str, extra_tokens: Optional[int] = None):
        super().__init__(message)
        self.extra_tokens = extra_tokens


class BaseCompletionClient(ABC):
    @classmethod
    @abstractmethod
    def from_configuration(
        cls,
        vendor_configuration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "BaseCompletionClient":
        raise NotImplementedError


class PromptCompletionClient(BaseCompletionClient):
    @abstractmethod
    async def complete(
        self, prompts: List[str], max_tokens: int
    ) -> List[PromptResponse]:

        raise NotImplementedError


class PromptCompletionWithLogitsClient(BaseCompletionClient):
    @abstractmethod
    async def complete(
        self, prompts: List[str], max_tokens: int
    ) -> List[PromptResponse]:

        raise NotImplementedError


class ChatClientRequest(TypedDict):
    conversation: BotConversation
    max_tokens: int
    functions: NotRequired[List[Dict]]
    tools: NotRequired[List[Dict]]
    tool_choice: NotRequired[str]


class ChatCompletionClient(BaseCompletionClient):
    @abstractmethod
    @overload
    async def complete(  # type: ignore
        self, request: ChatClientRequest, stream: Literal[False] = False
    ) -> ChatResponse:
        pass

    @abstractmethod
    @overload
    async def complete(
        self, request: ChatClientRequest, stream: Literal[True] = True
    ) -> AsyncIterator[ChatResponse]:
        pass

    @abstractmethod
    @overload
    async def complete(
        self, request: ChatClientRequest, stream: bool = False
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]:
        pass

    @abstractmethod
    async def complete(
        self,
        request: ChatClientRequest,
        stream: Union[Literal[True, False], bool] = False,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]:

        raise NotImplementedError
