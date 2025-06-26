import enum
from typing import AsyncIterator, Callable, Dict, List, Optional, Tuple, Union, overload

from typing_extensions import Literal
from zav.llm_domain import LLMClientConfiguration
from zav.llm_tracing import Span
from zav.prompt_completion import (
    BotConversation,
    ChatClientFactory,
    ChatClientRequest,
    ChatCompletionClient,
)
from zav.prompt_completion import ChatMessage as PcChatMessage
from zav.prompt_completion import (
    ChatMessageSender,
)
from zav.prompt_completion import FunctionCallRequest as PcFunctionCallRequest
from zav.prompt_completion import FunctionCallResponse as PcFunctionCallResponse
from zav.prompt_completion import ToolCallRequest as PcToolCallRequest
from zav.prompt_completion import ToolCallResponse as PcToolCallResponse
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage, FunctionCallRequest
from zav.agents_sdk.domain.tools import ToolsRegistry


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


class ChatCompletion(BaseModel):
    sender: ChatCompletionSender
    content: str
    image_uri: Optional[str] = None
    function_call_request: Optional[FunctionCallRequest] = None
    function_call_response: Optional[FunctionCallResponse] = None
    tool_call_requests: Optional[List[ToolCallRequest]] = None
    tool_call_responses: Optional[List[ToolCallResponse]] = None

    @classmethod
    def from_chat_message(cls, chat_message: ChatMessage):
        content = (
            "".join([part.text for part in chat_message.content_parts if part.text])
            if chat_message.content_parts
            else chat_message.content
        )
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
        )

    class Config:
        orm_mode = True


class ChatResponse(BaseModel):
    error: Optional[Exception]
    chat_completion: Optional[ChatCompletion]

    class Config:
        arbitrary_types_allowed = True


def parse_chat_message(message: Union[ChatMessage, ChatCompletion]) -> PcChatMessage:
    return PcChatMessage(
        sender=ChatMessageSender(message.sender),
        content=(
            "".join([part.text for part in message.content_parts if part.text])
            if isinstance(message, ChatMessage) and message.content_parts
            else message.content
        ),
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
    )


ShouldReturnToUser = bool


async def execute_tool_call_request(
    tools_registry: ToolsRegistry,
    tool_call_requests: Optional[List[ToolCallRequest]] = None,
    log_fn: Optional[Callable] = None,
    span: Optional[Span] = None,
) -> Tuple[Optional[ChatCompletion], ShouldReturnToUser]:
    if not tool_call_requests:
        return None, False
    if log_fn:
        log_fn({"Tool Call Requests": tool_call_requests})

    tool_outputs: List = []
    for tool_call_request in tool_call_requests:
        tool_call_id = tool_call_request.id
        function_call_request = tool_call_request.function_call_request
        new_span = (
            span.new(
                name=function_call_request.name,
                attributes={
                    "metadata": {"tool_call_id": tool_call_id},
                    "input": function_call_request.params or {},
                },
            )
            if span
            else None
        )
        try:
            exec_response = await tools_registry.execute(
                name=function_call_request.name, params=function_call_request.params
            )
            tool_response = str(exec_response)
            if new_span:
                new_span.end(attributes={"output": tool_response})
            if isinstance(exec_response, ChatCompletion):
                return exec_response, True

            tool_outputs.append({"id": tool_call_id, "response": tool_response})
        except ValueError as ve:
            str_value_err = str(ve)
            tool_outputs.append({"id": tool_call_id, "response": str_value_err})
            if new_span:
                new_span.end(attributes={"output": str_value_err})
            continue
        except Exception as e:
            str_err = str(e)
            if log_fn:
                log_fn({"Error": str_err})
            if new_span:
                new_span.end(attributes={"output": str_err})
            tool_outputs.append({"id": tool_call_id, "response": str_err})

    return (
        ChatCompletion(
            sender=ChatCompletionSender.TOOL,
            content="",
            tool_call_responses=[
                ToolCallResponse(
                    id=tool_output["id"],
                    response=tool_output["response"],
                )
                for tool_output in tool_outputs
            ],
        ),
        False,
    )


class ZAVChatCompletionClient:
    def __init__(
        self,
        chat_completion_client: ChatCompletionClient,
        span: Optional[Span] = None,
    ) -> None:
        self.__chat_completion_client = chat_completion_client
        self.__span = span

    @overload
    async def complete(  # type: ignore
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: int = 2048,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Literal[False] = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: int = 10,
    ) -> ChatResponse: ...

    @overload
    async def complete(
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: int = 2048,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Literal[True] = True,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: int = 10,
    ) -> AsyncIterator[ChatResponse]: ...

    @overload
    async def complete(
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: int = 2048,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: int = 10,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]: ...

    async def complete(
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: int = 2048,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Union[Literal[True, False], bool] = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: int = 20,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]:
        if max_nesting_level == 0:
            error_response = ChatResponse(
                error=Exception("Max nesting level reached."),
                chat_completion=None,
            )
            if stream:

                async def stream_error_response():
                    yield error_response

                return stream_error_response()
            else:
                return error_response
        if isinstance(tools, ToolsRegistry):
            tools_dict: Optional[List[Dict]] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.get_parameters_spec(),
                    },
                }
                for tool in tools.tools_index.values()
            ]
        else:
            tools_dict = tools

        req = ChatClientRequest(
            conversation=BotConversation(
                bot_setup_description=bot_setup_description,
                messages=(
                    [parse_chat_message(message) for message in messages]
                    if messages is not None
                    else (
                        [parse_chat_message(completion) for completion in completions]
                        if completions is not None
                        else []
                    )
                ),
            ),
            max_tokens=max_tokens,
            **({"functions": functions} if functions is not None else {}),
            **({"tools": tools_dict} if tools_dict is not None else {}),
            **({"tool_choice": tool_choice} if tool_choice is not None else {}),
        )

        chat_response = await self.__chat_completion_client.complete(
            request=req, stream=stream
        )
        if isinstance(chat_response, AsyncIterator):

            async def stream_response_chunks(chat_response):
                async for chat_response_chunk in chat_response:
                    response = self.__convert(chat_response_chunk)
                    if stream_tool_calls:
                        yield response
                    tool_completion, should_return_to_user = (
                        await execute_tool_call_request(
                            tools_registry=tools,
                            tool_call_requests=response.chat_completion.tool_call_requests,  # noqa: E501
                            log_fn=log_fn,
                            span=self.__span,
                        )
                        if (
                            response.chat_completion
                            and execute_tools
                            and isinstance(tools, ToolsRegistry)
                        )
                        else (None, False)
                    )
                    if tool_completion and stream_tool_calls:
                        yield ChatResponse(
                            error=None,
                            chat_completion=tool_completion,
                        )

                    if should_return_to_user:
                        break

                    inner_response_iterator = await self.__parse_inner_response(
                        response=response,
                        messages=messages,
                        completions=completions,
                        tool_completion=tool_completion,
                        max_tokens=max_tokens,
                        bot_setup_description=bot_setup_description,
                        tools=tools,
                        tool_choice=tool_choice,
                        stream=True,
                        execute_tools=execute_tools,
                        stream_tool_calls=stream_tool_calls,
                        log_fn=log_fn,
                        max_nesting_level=max_nesting_level,
                    )
                    if inner_response_iterator:
                        async for inner_response in inner_response_iterator:
                            yield inner_response
                    else:
                        yield response

            return stream_response_chunks(chat_response)
        else:
            response = self.__convert(chat_response)
            tool_completion, should_return_to_user = (
                await execute_tool_call_request(
                    tools_registry=tools,
                    tool_call_requests=response.chat_completion.tool_call_requests,
                    log_fn=log_fn,
                    span=self.__span,
                )
                if (
                    response.chat_completion
                    and execute_tools
                    and isinstance(tools, ToolsRegistry)
                )
                else (None, False)
            )
            if should_return_to_user:
                return ChatResponse(
                    error=None,
                    chat_completion=tool_completion,
                )
            inner_response = await self.__parse_inner_response(
                response=response,
                messages=messages,
                completions=completions,
                tool_completion=tool_completion,
                max_tokens=max_tokens,
                bot_setup_description=bot_setup_description,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
                execute_tools=execute_tools,
                stream_tool_calls=stream_tool_calls,
                log_fn=log_fn,
                max_nesting_level=max_nesting_level,
            )
            if inner_response:
                return inner_response
            else:
                return response

    def __convert(self, chat_response) -> ChatResponse:
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
                        function_call_request := (
                            chat_response_message.function_call_request
                        )
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
                    if (
                        tool_call_responses := chat_response_message.tool_call_responses
                    )
                    else None
                ),
            )
            if chat_response_message
            else None
        )
        return ChatResponse(
            error=chat_response.error,
            chat_completion=chat_completion,
        )

    @overload
    async def __parse_inner_response(  # type: ignore
        self,
        response: ChatResponse,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        tool_completion: Optional[ChatCompletion] = None,
        max_tokens: int = 2048,
        bot_setup_description: Optional[str] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Literal[False] = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: int = 10,
    ) -> Optional[ChatResponse]: ...

    @overload
    async def __parse_inner_response(
        self,
        response: ChatResponse,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        tool_completion: Optional[ChatCompletion] = None,
        max_tokens: int = 2048,
        bot_setup_description: Optional[str] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Literal[True] = True,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: int = 10,
    ) -> Optional[AsyncIterator[ChatResponse]]: ...

    @overload
    async def __parse_inner_response(
        self,
        response: ChatResponse,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        tool_completion: Optional[ChatCompletion] = None,
        max_tokens: int = 2048,
        bot_setup_description: Optional[str] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: int = 10,
    ) -> Optional[Union[AsyncIterator[ChatResponse], ChatResponse]]: ...

    async def __parse_inner_response(
        self,
        response: ChatResponse,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        tool_completion: Optional[ChatCompletion] = None,
        max_tokens: int = 2048,
        bot_setup_description: Optional[str] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Union[Literal[True, False], bool] = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: int = 10,
    ) -> Optional[Union[AsyncIterator[ChatResponse], ChatResponse]]:

        if tool_completion and response.chat_completion:
            inner_response = await self.complete(
                completions=(
                    [ChatCompletion.from_chat_message(message) for message in messages]
                    if messages
                    else completions or []
                )
                + [response.chat_completion, tool_completion],
                max_tokens=max_tokens,
                bot_setup_description=bot_setup_description,
                tools=tools,
                tool_choice=tool_choice if tool_choice != "required" else "auto",
                stream=stream,
                execute_tools=execute_tools,
                stream_tool_calls=stream_tool_calls,
                log_fn=log_fn,
                max_nesting_level=max_nesting_level - 1,
            )
            return inner_response
        else:
            return None


class ZAVChatCompletionClientFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls, config: LLMClientConfiguration, span: Optional[Span] = None
    ) -> ZAVChatCompletionClient:
        chat_completion_client = ChatClientFactory.create(config, span=span)
        return ZAVChatCompletionClient(chat_completion_client, span=span)
