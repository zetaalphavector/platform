import asyncio
import json
import re
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Union, cast, overload

import openai
from openai import BadRequestError
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_named_tool_choice_param import (
    ChatCompletionNamedToolChoiceParam,
)
from openai.types.chat.chat_completion_named_tool_choice_param import (
    Function as ToolChoiceFunction,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.completion_create_params import Function
from openai.types.completion_choice import CompletionChoice

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore

from typing_extensions import Literal
from zav.llm_domain import (
    LLMModelConfiguration,
    LLMModelType,
    LLMProviderName,
    OpenAIConfiguration,
)
from zav.llm_tracing import Span, now

from zav.prompt_completion.adapters.tracing import create_span, end_span
from zav.prompt_completion.client import (
    BotConversation,
    ChatClientRequest,
    ChatCompletionClient,
    ChatMessage,
    ChatMessageSender,
    ChatResponse,
    FunctionCallRequest,
    PromptAnswer,
    PromptAnswerWithLogits,
    PromptCompletionClient,
    PromptCompletionWithLogitsClient,
    PromptResponse,
    PromptTooLargeError,
    ToolCallRequest,
)
from zav.prompt_completion.client_factories import (
    ChatClientFactory,
    PromptClientFactory,
    PromptWithLogitsClientFactory,
)


class OAIFunctionCall(BaseModel):
    name: str
    arguments: str


class OAIToolCall(BaseModel):
    id: str
    function: Optional[OAIFunctionCall] = None


def __extract_int_from_text(pattern: str, text: str) -> Optional[int]:
    """Extract int from text."""
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None


def generate_prompt_too_long_error(error_message: str):
    max_tokens_limit = __extract_int_from_text(
        r"maximum context length is (\d+) tokens", error_message
    )
    requested_tokens = __extract_int_from_text(
        r"resulted in (\d+) tokens", error_message
    )
    extra_tokens = (
        requested_tokens - max_tokens_limit
        if max_tokens_limit and requested_tokens
        else None
    )
    return PromptTooLargeError(error_message, extra_tokens)


def build_client(vendor_configuration: OpenAIConfiguration):
    organization = vendor_configuration.openai_org.get_unencrypted_secret()
    api_key = vendor_configuration.openai_api_key.get_unencrypted_secret()
    if vendor_configuration.openai_api_type == "azure":
        if vendor_configuration.openai_api_base:
            return openai.AsyncAzureOpenAI(
                api_key=api_key,
                organization=organization,
                api_version=vendor_configuration.openai_api_version,
                azure_endpoint=vendor_configuration.openai_api_base,
            )
        else:
            return openai.AsyncAzureOpenAI(
                api_key=api_key,
                organization=organization,
                api_version=vendor_configuration.openai_api_version,
            )
    else:
        return openai.AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=vendor_configuration.openai_api_base,
        )


@PromptWithLogitsClientFactory.register(
    LLMProviderName.OPENAI, LLMModelType.PROMPT_WITH_LOGITS
)
class OpenAiPromptWithLogitsClient(PromptCompletionWithLogitsClient):
    __INCLUDE_LOGPROBS_FOR_MOST_LIKELY_TOKEN = 0

    def __init__(
        self,
        client: Union[openai.AsyncAzureOpenAI, openai.AsyncOpenAI],
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ):
        self.__client = client
        self.__model_name = model_configuration.name
        self.__model_temperature = model_configuration.temperature
        self.__span = span

    async def complete(
        self, prompts: List[str], max_tokens: int
    ) -> List[PromptResponse]:
        return await asyncio.gather(
            *[self.__complete_prompt(prompt, max_tokens) for prompt in prompts]
        )

    async def __complete_prompt(self, prompt: str, max_tokens: int) -> PromptResponse:
        try:
            generation_span = (
                self.__span.new(
                    name="prompt-completion",
                    attributes={
                        "observation_type": "generation",
                        "model": self.__model_name,
                        "input": prompt,
                        "model_parameters": {
                            "temperature": self.__model_temperature,
                            "max_tokens": max_tokens,
                        },
                    },
                )
                if self.__span
                else None
            )
            answer = await self.__client.completions.create(
                model=self.__model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.__model_temperature,
                logprobs=self.__INCLUDE_LOGPROBS_FOR_MOST_LIKELY_TOKEN,
            )
            if generation_span:
                generation_span.end(
                    attributes={
                        "output": answer.choices[0].text,
                        **(
                            {
                                "usage": {
                                    "input": answer.usage.prompt_tokens,
                                    "output": answer.usage.completion_tokens,
                                    "total": answer.usage.total_tokens,
                                    "unit": "TOKENS",
                                }
                            }
                            if answer.usage
                            else {}
                        ),
                    }
                )
            return self.__prompt_response_from(answer.choices[0])
        except BadRequestError as e:
            if generation_span:
                generation_span.end(
                    attributes={
                        "level": "ERROR",
                        "status_message": e.message,
                    }
                )
            # Same error for any number of prompts, the message refers to the
            # first prompt that was too long. It doesn't say which one.
            if e.status_code == 400 and (
                "context_length_exceeded" in e.message or "string too long" in e.message
            ):
                error = generate_prompt_too_long_error(e.message)
            else:
                error = Exception(f"Prompt completion failed with error: {e.message}")
            return PromptResponse(error=error, prompt_answer=None)
        except Exception as e:
            if generation_span:
                generation_span.end(
                    attributes={
                        "level": "ERROR",
                        "status_message": str(e),
                    }
                )
            return PromptResponse(error=e, prompt_answer=None)

    def __prompt_response_from(self, answer_choice: CompletionChoice) -> PromptResponse:
        if (
            not answer_choice.logprobs
            or not answer_choice.logprobs.token_logprobs
            or not answer_choice.logprobs.tokens
        ):
            return PromptResponse(
                error=Exception("No logprobs returned"), prompt_answer=None
            )

        return PromptResponse(
            error=None,
            prompt_answer=PromptAnswerWithLogits(
                text=answer_choice.text.strip(),
                token_scores=[
                    {token: token_logprob}
                    for token, token_logprob in zip(
                        answer_choice.logprobs.tokens,
                        answer_choice.logprobs.token_logprobs,
                    )
                ],
            ),
        )

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: OpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "OpenAiPromptWithLogitsClient":
        client = build_client(vendor_configuration)
        return cls(client=client, model_configuration=model_configuration, span=span)


@PromptClientFactory.register(LLMProviderName.OPENAI, LLMModelType.PROMPT_WITH_LOGITS)
@PromptClientFactory.register(LLMProviderName.OPENAI, LLMModelType.PROMPT)
class OpenAiPromptClient(PromptCompletionClient):
    def __init__(
        self,
        client: Union[openai.AsyncAzureOpenAI, openai.AsyncOpenAI],
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ):
        self.__client = client
        self.__model_name = model_configuration.name
        self.__model_temperature = model_configuration.temperature
        self.__span = span

    async def complete(
        self, prompts: List[str], max_tokens: int
    ) -> List[PromptResponse]:
        return await asyncio.gather(
            *[self.__complete_prompt(prompt, max_tokens) for prompt in prompts]
        )

    async def __complete_prompt(self, prompt: str, max_tokens: int) -> PromptResponse:
        try:
            generation_span = (
                self.__span.new(
                    name="prompt-completion",
                    attributes={
                        "observation_type": "generation",
                        "model": self.__model_name,
                        "input": prompt,
                        "model_parameters": {
                            "temperature": self.__model_temperature,
                            "max_tokens": max_tokens,
                        },
                    },
                )
                if self.__span
                else None
            )
            answer = await self.__client.completions.create(
                model=self.__model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.__model_temperature,
            )
            if generation_span:
                generation_span.end(
                    attributes={
                        "output": answer.choices[0].text,
                        **(
                            {
                                "usage": {
                                    "input": answer.usage.prompt_tokens,
                                    "output": answer.usage.completion_tokens,
                                    "total": answer.usage.total_tokens,
                                    "unit": "TOKENS",
                                }
                            }
                            if answer.usage
                            else {}
                        ),
                    }
                )
            prompt_answer = PromptAnswer(text=answer.choices[0].text.strip())
            return PromptResponse(error=None, prompt_answer=prompt_answer)
        except BadRequestError as e:
            if generation_span:
                generation_span.end(
                    attributes={
                        "level": "ERROR",
                        "status_message": e.message,
                    }
                )
            # Same error for any number of prompts, the message refers to the
            # first prompt that was too long. It doesn't say which one.
            if e.status_code == 400 and (
                "context_length_exceeded" in e.message or "string too long" in e.message
            ):
                error = generate_prompt_too_long_error(e.message)
                return PromptResponse(error=error, prompt_answer=None)
            else:
                return PromptResponse(error=e, prompt_answer=None)
        except Exception as e:
            if generation_span:
                generation_span.end(
                    attributes={
                        "level": "ERROR",
                        "status_message": str(e),
                    }
                )
            return PromptResponse(error=e, prompt_answer=None)

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: OpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "OpenAiPromptClient":
        client = build_client(vendor_configuration)
        return cls(client=client, model_configuration=model_configuration, span=span)


@ChatClientFactory.register(LLMProviderName.OPENAI, LLMModelType.CHAT)
class OpenAiChatClient(ChatCompletionClient):
    TOKENS_PER_CHARACTER = 4
    __SENDER_TO_ROLE = {
        ChatMessageSender.BOT: "assistant",
        ChatMessageSender.USER: "user",
        ChatMessageSender.FUNCTION: "function",
        ChatMessageSender.TOOL: "tool",
    }
    __ROLE_TO_SENDER = {
        "assistant": ChatMessageSender.BOT,
        "user": ChatMessageSender.USER,
        "function": ChatMessageSender.FUNCTION,
        "tool": ChatMessageSender.TOOL,
    }

    def __init__(
        self,
        client: Union[openai.AsyncAzureOpenAI, openai.AsyncOpenAI],
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ):
        self.__client = client
        self.__model_name = model_configuration.name
        self.__model_temperature = model_configuration.temperature
        self.__json_output = model_configuration.json_output
        self.__interleave_system_message = model_configuration.interleave_system_message
        self.__span = span

    @overload
    async def complete(  # type: ignore
        self,
        request: ChatClientRequest,
        stream: Literal[False] = False,
    ) -> ChatResponse: ...

    @overload
    async def complete(
        self,
        request: ChatClientRequest,
        stream: Literal[True] = True,
    ) -> AsyncIterator[ChatResponse]: ...

    @overload
    async def complete(
        self,
        request: ChatClientRequest,
        stream: bool = False,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]: ...

    async def complete(
        self,
        request: ChatClientRequest,
        stream: Union[Literal[True, False], bool] = False,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]:
        try:
            messages = self.__messages_from(request["conversation"])
            functions_dict = (
                {"functions": [cast(Function, fn) for fn in functions]}
                if (functions := request.get("functions")) is not None
                else {}
            )
            tools_dict: Dict = (
                {
                    "tools": [cast(ChatCompletionToolParam, tool) for tool in tools],
                }
                if (tools := request.get("tools"))
                else {}
            )

            generation_span = create_span(
                messages=messages,
                functions_dict=functions_dict,
                tools_dict=tools_dict,
                model_name=self.__model_name,
                model_temperature=self.__model_temperature,
                span=self.__span,
                max_tokens=request["max_tokens"],
                json_output=self.__json_output,
                interleave_system_message=self.__interleave_system_message,
                stream=stream,
            )
            if tools_dict:
                tools_dict["tool_choice"] = (
                    ChatCompletionNamedToolChoiceParam(
                        type="function",
                        function=ToolChoiceFunction(name=tool_choice),
                    )
                    if (
                        (tool_choice := request.get("tool_choice"))
                        and tool_choice not in ["auto", "none"]
                    )
                    else request.get("tool_choice", "auto")
                )
            response = await self.__client.chat.completions.create(
                model=self.__model_name,
                messages=messages,
                max_tokens=request["max_tokens"],
                temperature=self.__model_temperature,
                stream=stream,
                **functions_dict,
                **(
                    {"response_format": {"type": "json_object"}}
                    if self.__json_output
                    else {}
                ),  # type: ignore
                **tools_dict,
            )
            if isinstance(response, AsyncIterator):

                async def stream_response(
                    response: AsyncIterator[ChatCompletionChunk],
                ) -> AsyncIterator[ChatResponse]:
                    completion_start_time: Optional[datetime] = None
                    content_buffer: Optional[str] = None
                    role_buffer: Optional[str] = None
                    function_call_buffer: Optional[OAIFunctionCall] = None
                    function_call_buffer_trace: Optional[OAIFunctionCall] = None
                    tool_calls_buffer: List[OAIToolCall] = []
                    tool_calls_buffer_trace: List[OAIToolCall] = []
                    usage_buffer: Dict = {}
                    async for chunk in response:
                        if generation_span and completion_start_time is None:
                            completion_start_time = now()
                            generation_span.update(
                                attributes={
                                    "completion_start_time": completion_start_time,
                                }
                            )
                        if generation_span:
                            usage_buffer = (
                                {
                                    "usage": {
                                        "input": chunk.usage.prompt_tokens,
                                        "output": chunk.usage.completion_tokens,
                                        "total": chunk.usage.total_tokens,
                                        "unit": "TOKENS",
                                    }
                                }
                                if chunk.usage
                                else {}
                            )
                        choice_chunk = chunk.choices[0] if chunk.choices else None
                        if choice_chunk is None:
                            # This is a completion chunk with no choices, skip it
                            continue

                        if choice_chunk.delta.role is not None:
                            role_buffer = choice_chunk.delta.role

                        if (fn_call := choice_chunk.delta.function_call) is not None:
                            if fn_call.name is not None:
                                function_call_buffer = OAIFunctionCall(
                                    name=fn_call.name,
                                    arguments=fn_call.arguments or "",
                                )
                            if fn_call.arguments is not None and function_call_buffer:
                                function_call_buffer.arguments += fn_call.arguments
                            # We need to wait until the function call is complete
                            # because we don't support non-parseable arguments
                            continue
                        if function_call_buffer:
                            yield ChatResponse(
                                error=None,
                                chat_message=self.__parse_chat_message(
                                    content=content_buffer,
                                    role=role_buffer,
                                    function_call=function_call_buffer,
                                    tool_calls=None,
                                ),
                            )
                            function_call_buffer_trace = function_call_buffer
                            function_call_buffer = None

                        if (tool_calls := choice_chunk.delta.tool_calls) is not None:
                            for tool_call in tool_calls:
                                if tool_call.id is not None:
                                    tool_calls_buffer.append(
                                        OAIToolCall(id=tool_call.id)
                                    )
                                if (
                                    tool_fn := tool_call.function
                                ) is not None and tool_calls_buffer:
                                    existing_tool_call = tool_calls_buffer[-1]
                                    if tool_fn.name is not None:
                                        existing_tool_call.function = OAIFunctionCall(
                                            name=tool_fn.name,
                                            arguments=tool_fn.arguments or "",
                                        )
                                    if (
                                        tool_fn.arguments is not None
                                        and existing_tool_call.function is not None
                                    ):
                                        existing_tool_call.function.arguments += (
                                            tool_fn.arguments
                                        )
                            # We need to wait until all tool calls are complete
                            # because we don't support non-parseable arguments
                            continue

                        if tool_calls_buffer:
                            yield ChatResponse(
                                error=None,
                                chat_message=self.__parse_chat_message(
                                    content=content_buffer,
                                    role=role_buffer,
                                    function_call=None,
                                    tool_calls=tool_calls_buffer,
                                ),
                            )
                            tool_calls_buffer_trace = tool_calls_buffer
                            tool_calls_buffer = []

                        if choice_chunk.delta.content:
                            content_buffer = (
                                content_buffer + choice_chunk.delta.content
                                if content_buffer
                                else choice_chunk.delta.content
                            )
                            yield ChatResponse(
                                error=None,
                                chat_message=self.__parse_chat_message(
                                    content=content_buffer,
                                    role=role_buffer,
                                    function_call=None,
                                    tool_calls=None,
                                ),
                            )
                    end_span(
                        tool_calls=tool_calls_buffer_trace,
                        usage=usage_buffer,
                        span=generation_span,
                        content=content_buffer,
                        role=role_buffer,
                        function_call=function_call_buffer_trace,
                    )

                return stream_response(response)

            else:
                choice = response.choices[0]
                function_call = (
                    OAIFunctionCall(
                        name=fn_call.name,
                        arguments=fn_call.arguments,
                    )
                    if (fn_call := choice.message.function_call)
                    else None
                )
                tool_calls = (
                    [
                        OAIToolCall(
                            id=tool_call.id,
                            function=OAIFunctionCall(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                        )
                        for tool_call in tool_calls
                    ]
                    if (tool_calls := choice.message.tool_calls)
                    else []
                )
                end_span(
                    tool_calls=tool_calls,
                    usage=(
                        {
                            "usage": {
                                "input": response.usage.prompt_tokens,
                                "output": response.usage.completion_tokens,
                                "total": response.usage.total_tokens,
                                "unit": "TOKENS",
                            }
                        }
                        if response.usage
                        else {}
                    ),
                    span=generation_span,
                    content=choice.message.content,
                    role=choice.message.role,
                    function_call=function_call,
                )
                chat_message = self.__parse_chat_message(
                    content=choice.message.content,
                    role=choice.message.role,
                    function_call=function_call,
                    tool_calls=tool_calls,
                )
                return ChatResponse(error=None, chat_message=chat_message)
        except BadRequestError as e:
            if generation_span:
                generation_span.end(
                    attributes={
                        "level": "ERROR",
                        "status_message": e.message,
                    }
                )
            if e.status_code == 400 and (
                "context_length_exceeded" in e.message or "string too long" in e.message
            ):
                error = generate_prompt_too_long_error(e.message)
            else:
                error = e
            return (
                stream_response_item(ChatResponse(error=error, chat_message=None))
                if stream
                else ChatResponse(error=error, chat_message=None)
            )
        except Exception as error:
            if generation_span:
                generation_span.end(
                    attributes={
                        "level": "ERROR",
                        "status_message": str(error),
                    }
                )
            return (
                stream_response_item(ChatResponse(error=error, chat_message=None))
                if stream
                else ChatResponse(error=error, chat_message=None)
            )

    def __messages_from(self, conversation: BotConversation):
        messages: List[Any] = []
        for message in conversation.messages:
            if message.function_call_response:
                messages.append(
                    {
                        "role": self.__SENDER_TO_ROLE[message.sender],
                        "name": message.function_call_response.function_name,
                        "content": message.function_call_response.function_response,
                    }
                )
            elif message.function_call_request:
                messages.append(
                    {
                        "role": self.__SENDER_TO_ROLE[message.sender],
                        "content": None,  # If we pass the content here it gets confused
                        "function_call": {
                            "name": message.function_call_request.function_name,
                            "arguments": json.dumps(
                                message.function_call_request.function_params
                            ),
                        },
                    }
                )
            elif message.tool_call_responses:
                for tool_call_response in message.tool_call_responses:
                    messages.append(
                        {
                            "role": self.__SENDER_TO_ROLE[message.sender],
                            "content": tool_call_response.tool_response,  # noqa: E501
                            "tool_call_id": tool_call_response.id,
                        }
                    )
            elif message.tool_call_requests:
                messages.append(
                    {
                        "role": self.__SENDER_TO_ROLE[message.sender],
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_request.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call_request.function_call_request.function_name,  # noqa: E501
                                    "arguments": json.dumps(
                                        tool_call_request.function_call_request.function_params  # noqa: E501
                                    ),
                                },
                            }
                            for tool_call_request in message.tool_call_requests
                        ],
                    }
                )

            else:
                if message.sender == ChatMessageSender.BOT:
                    messages.append(
                        {
                            "role": self.__SENDER_TO_ROLE[message.sender],
                            "content": message.content,
                        }
                    )
                elif message.sender == ChatMessageSender.USER:
                    content: List[Dict[str, Any]] = []
                    if message.content:
                        content.append(
                            {
                                "type": "text",
                                "text": message.content,
                            }
                        )
                    if message.image_uri:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": message.image_uri},
                            }
                        )
                    if len(content) > 0:
                        messages.append(
                            {
                                "role": self.__SENDER_TO_ROLE[message.sender],
                                "content": content,
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": self.__SENDER_TO_ROLE[message.sender],
                                "content": "",
                            }
                        )

        if conversation.bot_setup_description is not None:
            system_setup_message = {
                "role": "system",
                "content": conversation.bot_setup_description,
            }

            messages = [system_setup_message] + messages
            if self.__interleave_system_message == "repeat_before_last_user_message":
                last_user_message_index: Optional[int] = next(
                    (
                        len(messages) - 1 - i
                        for i, message in enumerate(reversed(messages))
                        if message["role"] == "user"
                    ),
                    None,
                )
                if last_user_message_index is not None:
                    messages = (
                        messages[:last_user_message_index]
                        + [system_setup_message]
                        + messages[last_user_message_index:]
                    )

        return messages

    @staticmethod
    def __parse_partial_json(string) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(string)
        except json.JSONDecodeError:
            pass

        # Initialize variables.
        new_s = ""
        stack = []
        is_inside_string = False
        escaped = False

        # Process each character in the string one at a time.
        for char in string:
            if is_inside_string:
                if char == '"' and not escaped:
                    is_inside_string = False
                elif char == "\n" and not escaped:
                    char = (
                        "\\n"  # Replace the newline character with the escape sequence.
                    )
                elif char == "\\":
                    escaped = not escaped
                else:
                    escaped = False
            else:
                if char == '"':
                    is_inside_string = True
                    escaped = False
                elif char == "{":
                    stack.append("}")
                elif char == "[":
                    stack.append("]")
                elif char == "}" or char == "]":
                    if stack and stack[-1] == char:
                        stack.pop()
                    else:
                        # Mismatched closing character; the input is malformed.
                        return None

            # Append the processed character to the new string.
            new_s += char

        # If we're still inside a string at the end of processing, we need to close the
        # string.
        if is_inside_string:
            new_s += '"'

        # Close any remaining open structures in the reverse order that they were open.
        for closing_char in reversed(stack):
            new_s += closing_char

        # Attempt to parse the modified string as JSON.
        try:
            return json.loads(new_s)
        except json.JSONDecodeError:
            # If we still can't parse the string as JSON, return None.
            return None

    def __parse_chat_message(
        self,
        content: Optional[str],
        role: Optional[str],
        function_call: Optional[OAIFunctionCall],
        tool_calls: Optional[List[OAIToolCall]],
    ):
        msg = ChatMessage(
            content=content or "",
            sender=self.__ROLE_TO_SENDER[role or "assistant"],
        )
        if function_call and function_call.name:
            msg.function_call_request = FunctionCallRequest(
                function_name=function_call.name,
                function_params=(
                    self.__parse_partial_json(function_call.arguments)
                    if function_call.arguments
                    else None
                ),
            )
        if tool_calls:
            msg.tool_call_requests = [
                ToolCallRequest(
                    id=tool_call.id,
                    function_call_request=FunctionCallRequest(
                        function_name=tool_call.function.name,
                        function_params=(
                            self.__parse_partial_json(tool_call.function.arguments)
                            if tool_call.function.arguments
                            else None
                        ),
                    ),
                )
                for tool_call in tool_calls
                if tool_call.id and tool_call.function and tool_call.function.name
            ]
        return msg

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: OpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "OpenAiChatClient":
        client = build_client(vendor_configuration)
        return cls(client=client, model_configuration=model_configuration, span=span)


@PromptClientFactory.register(LLMProviderName.OPENAI, LLMModelType.CHAT)
class OpenAiChatClient2PromptClientAdapter(PromptCompletionClient):
    def __init__(self, chat_client: OpenAiChatClient):
        self.__chat_client = chat_client

    async def complete(
        self,
        prompts: List[str],
        max_tokens: int,
    ) -> List[PromptResponse]:
        bot_conversations = [self.__to_bot_conversation(prompt) for prompt in prompts]

        chat_responses = await asyncio.gather(
            *[
                self.__chat_client.complete(
                    ChatClientRequest(conversation=conversation, max_tokens=max_tokens),
                )
                for conversation in bot_conversations
            ]
        )
        return [
            self.__to_prompt_response(chat_response) for chat_response in chat_responses
        ]

    def __to_bot_conversation(self, prompt: str):
        return BotConversation(
            messages=[
                ChatMessage(
                    content=prompt,
                    sender=ChatMessageSender.USER,
                )
            ],
            bot_setup_description=None,
        )

    def __to_prompt_response(self, chat_response: ChatResponse) -> PromptResponse:
        prompt_answer = (
            PromptAnswer(text=chat.content)
            if (chat := chat_response.chat_message)
            else None
        )
        return PromptResponse(error=chat_response.error, prompt_answer=prompt_answer)

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: OpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "OpenAiChatClient2PromptClientAdapter":
        chat_client = OpenAiChatClient.from_configuration(
            vendor_configuration, model_configuration, span=span
        )
        return cls(chat_client=chat_client)


async def stream_response_item(chat_response):
    yield chat_response
