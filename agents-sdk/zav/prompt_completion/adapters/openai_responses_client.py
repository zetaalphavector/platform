import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast, overload

import openai
from openai import BadRequestError, RateLimitError
from typing_extensions import AsyncIterator, Literal
from zav.llm_domain import (
    LLMModelConfiguration,
    LLMModelType,
    LLMProviderName,
    OpenAIConfiguration,
)
from zav.llm_tracing import Span
from zav.logging import logger

from zav.prompt_completion.adapters.openai_clients import (
    OpenAiChatClient2PromptClientAdapter,
    generate_prompt_too_long_error,
    stream_response_item,
)
from zav.prompt_completion.adapters.tracing import create_span, end_span
from zav.prompt_completion.client import (
    BotConversation,
    ChatClientRequest,
    ChatCompletionClient,
    ChatMessage,
    ChatMessageSender,
    ChatResponse,
    FunctionCallRequest,
    ToolCallRequest,
)
from zav.prompt_completion.client_factories import (
    ChatClientFactory,
    PromptClientFactory,
)


@ChatClientFactory.register(LLMProviderName.OPENAI, LLMModelType.RESPONSES)
class OpenAiResponsesClient(ChatCompletionClient):
    __SENDER_TO_ROLE = {
        ChatMessageSender.USER: "user",
        ChatMessageSender.BOT: "assistant",
        ChatMessageSender.DEVELOPER: "developer",
        ChatMessageSender.SYSTEM: "system",
    }

    def __init__(
        self,
        client: openai.AsyncOpenAI,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ):
        self.__client = client
        self.__model_configuration = model_configuration
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

    async def complete(  # noqa: C901
        self,
        request: ChatClientRequest,
        stream: Union[Literal[True, False], bool] = False,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]:
        generation_span = None
        try:
            legacy_tools, modern_tools = self.__tools_from(request)
            input_items = self.__input_from(conversation=request["conversation"])
            tools = [*legacy_tools.values(), *modern_tools.values()]
            max_tokens = request.get(
                "max_tokens", self.__model_configuration.max_tokens
            )
            generation_span = create_span(
                messages=self.__trace_messages(request["conversation"]),
                functions_dict=(
                    {"functions": request["functions"]}
                    if request.get("functions") is not None
                    else {}
                ),
                tools_dict=(
                    {"tools": request["tools"]}
                    if request.get("tools") is not None
                    else {}
                ),
                model_name=self.__model_configuration.name,
                model_temperature=self.__model_configuration.temperature,
                api_endpoint="responses",
                span=self.__span,
                max_tokens=max_tokens,
                json_output=self.__model_configuration.json_output,
                interleave_system_message=(
                    self.__model_configuration.interleave_system_message
                ),
                stream=stream,
            )

            kwargs: Dict[str, Any] = {
                "model": self.__model_configuration.name,
                "input": input_items,
                "store": False,
                "include": ["reasoning.encrypted_content"],
                "stream": stream,
            }
            instructions = request["conversation"].bot_setup_description
            if instructions is not None:
                kwargs["instructions"] = instructions
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = self.__tool_choice_from(request)
            if max_tokens is not None:
                kwargs["max_output_tokens"] = max_tokens
            if self.__model_configuration.temperature is not None:
                kwargs["temperature"] = self.__model_configuration.temperature
            parallel_tool_calls = request.get(
                "parallel_tool_calls",
                self.__model_configuration.parallel_tool_calls,
            )
            if legacy_tools:
                kwargs["parallel_tool_calls"] = False
            elif parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = parallel_tool_calls
            reasoning_effort = request.get(
                "reasoning_effort", self.__model_configuration.reasoning_effort
            )
            reasoning_options: Dict[str, Any] = {"summary": "auto"}
            if reasoning_effort is not None:
                reasoning_options["effort"] = reasoning_effort
            kwargs["reasoning"] = reasoning_options
            text: Dict[str, Any] = {}
            verbosity = request.get("verbosity", self.__model_configuration.verbosity)
            if verbosity is not None:
                text["verbosity"] = verbosity
            if self.__model_configuration.json_output:
                text["format"] = {"type": "json_object"}
            if text:
                kwargs["text"] = text

            response = await self.__client.responses.create(**kwargs)
            if stream:
                return self.__stream_response(
                    stream=cast(AsyncIterator[Any], response),
                    span=generation_span,
                    legacy_tool_names=set(legacy_tools),
                    modern_tool_names=set(modern_tools),
                )
            response_data = self.__response_dict(response)
            return self.__completed_response(
                response_data=response_data,
                span=generation_span,
                legacy_tool_names=set(legacy_tools),
                modern_tool_names=set(modern_tools),
            )
        except Exception as error:
            error_response = self.__error_response(error, generation_span)
            return stream_response_item(error_response) if stream else error_response

    def __tools_from(
        self, request: ChatClientRequest
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        legacy_tools: Dict[str, Dict[str, Any]] = {}
        modern_tools: Dict[str, Dict[str, Any]] = {}
        for function in request.get("functions", []):
            tool = self.__function_tool_from(function)
            name = cast(str, tool["name"])
            if name in legacy_tools:
                raise ValueError(f"Duplicate function name: {name}")
            legacy_tools[name] = tool
        for source_tool in request.get("tools", []):
            if source_tool.get("type") != "function" or not isinstance(
                source_tool.get("function"), dict
            ):
                raise ValueError("The Responses adapter only supports function tools.")
            tool = self.__function_tool_from(source_tool["function"])
            name = cast(str, tool["name"])
            if name in legacy_tools or name in modern_tools:
                raise ValueError(f"Duplicate function name: {name}")
            modern_tools[name] = tool
        return legacy_tools, modern_tools

    @staticmethod
    def __function_tool_from(function: Dict[str, Any]) -> Dict[str, Any]:
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Function tools require a non-empty name.")
        tool: Dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": function.get("parameters", {}),
            "strict": function.get("strict", False),
        }
        if function.get("description") is not None:
            tool["description"] = function["description"]
        return tool

    @staticmethod
    def __tool_choice_from(request: ChatClientRequest) -> Any:
        tool_choice = request.get("tool_choice", "auto")
        if tool_choice in ("auto", "none", "required"):
            return tool_choice
        return {"type": "function", "name": tool_choice}

    def __input_from(  # noqa: C901
        self,
        conversation: BotConversation,
    ) -> List[Dict[str, Any]]:
        input_items: List[Dict[str, Any]] = []
        modern_call_order: List[str] = []
        modern_call_ids: Set[str] = set()
        emitted_output_ids: Set[str] = set()
        pending_legacy_call: Optional[Tuple[str, str]] = None

        for message_index, message in enumerate(conversation.messages):
            if message.response_replay_items is not None:
                replay_calls: List[Tuple[str, str]] = []
                for item in message.response_replay_items:
                    if item.get("type") != "function_call":
                        continue
                    call_id = item.get("call_id")
                    name = item.get("name")
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError("Replayed function calls require a call_id.")
                    replay_calls.append((call_id, cast(str, name)))
                if message.function_call_request is not None:
                    if len(replay_calls) > 1:
                        raise ValueError(
                            "Legacy functions support one function call per response."
                        )
                    if replay_calls:
                        pending_legacy_call = replay_calls[0]
                else:
                    for call_id, _ in replay_calls:
                        if call_id in modern_call_ids:
                            raise ValueError(f"Duplicate tool call ID: {call_id}")
                        modern_call_ids.add(call_id)
                        modern_call_order.append(call_id)
                # status is an output-only Item field; the input schema rejects it.
                input_items.extend(
                    {key: value for key, value in item.items() if key != "status"}
                    for item in deepcopy(message.response_replay_items)
                )
                continue

            if message.function_call_request is not None:
                name = message.function_call_request.function_name
                call_id = self.__legacy_call_id(
                    message_index, message.function_call_request
                )
                pending_legacy_call = (call_id, name)
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": json.dumps(
                            message.function_call_request.function_params or {},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
                continue

            if message.function_call_response is not None:
                if pending_legacy_call is None:
                    raise ValueError(
                        "Legacy function output has no preceding function call."
                    )
                call_id, name = pending_legacy_call
                if message.function_call_response.function_name != name:
                    raise ValueError(
                        "Legacy function output does not match its function call."
                    )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": message.function_call_response.function_response
                        or "",
                    }
                )
                pending_legacy_call = None
                continue

            if message.tool_call_requests is not None:
                for tool_call in message.tool_call_requests:
                    name = tool_call.function_call_request.function_name
                    if tool_call.id in modern_call_ids:
                        raise ValueError(f"Duplicate tool call ID: {tool_call.id}")
                    modern_call_ids.add(tool_call.id)
                    modern_call_order.append(tool_call.id)
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.id,
                            "name": name,
                            "arguments": json.dumps(
                                tool_call.function_call_request.function_params or {},
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    )
                continue

            if message.tool_call_responses is not None:
                response_by_id = {
                    tool_response.id: tool_response
                    for tool_response in message.tool_call_responses
                }
                if len(response_by_id) != len(message.tool_call_responses):
                    raise ValueError("Duplicate tool output ID.")
                unknown_ids = set(response_by_id) - modern_call_ids
                if unknown_ids:
                    raise ValueError(
                        "Tool outputs have no matching calls: "
                        + ", ".join(sorted(unknown_ids))
                    )
                for call_id in modern_call_order:
                    if call_id not in response_by_id:
                        continue
                    if call_id in emitted_output_ids:
                        raise ValueError(f"Duplicate tool output ID: {call_id}")
                    tool_response = response_by_id[call_id]
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": tool_response.tool_response or "",
                        }
                    )
                    emitted_output_ids.add(call_id)
                continue

            input_items.append(self.__message_input_from(message))

        if (
            conversation.bot_setup_description is not None
            and self.__model_configuration.interleave_system_message
            == "repeat_before_last_user_message"
        ):
            last_user_index = next(
                (
                    len(input_items) - 1 - index
                    for index, item in enumerate(reversed(input_items))
                    if item.get("role") == "user"
                ),
                None,
            )
            if last_user_index is not None:
                input_items.insert(
                    last_user_index,
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": conversation.bot_setup_description,
                            }
                        ],
                    },
                )
        return input_items

    def __message_input_from(self, message: ChatMessage) -> Dict[str, Any]:
        role = self.__SENDER_TO_ROLE.get(message.sender)
        if role is None:
            raise ValueError(
                f"Unsupported unstructured message sender: {message.sender.value}"
            )
        text_type = "output_text" if role == "assistant" else "input_text"
        content: List[Dict[str, Any]] = []
        if message.content_parts:
            for part in message.content_parts:
                if part.document is not None:
                    raise ValueError(
                        "The Responses adapter does not support document content."
                    )
                if part.text is not None:
                    content.append({"type": text_type, "text": part.text})
                if part.image is not None:
                    if role == "assistant":
                        raise ValueError(
                            "The Responses adapter does not support image content"
                            " on assistant messages."
                        )
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": part.image.image_uri,
                        }
                    )
        else:
            if message.content:
                content.append({"type": text_type, "text": message.content})
            if message.image_uri:
                if role == "assistant":
                    raise ValueError(
                        "The Responses adapter does not support image content"
                        " on assistant messages."
                    )
                content.append({"type": "input_image", "image_url": message.image_uri})
        return {"role": role, "content": content}

    @staticmethod
    def __legacy_call_id(
        message_index: int, function_call_request: FunctionCallRequest
    ) -> str:
        identity = json.dumps(
            {
                "message_index": message_index,
                "name": function_call_request.function_name,
                "arguments": function_call_request.function_params or {},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return f"call_legacy_{digest}"

    @staticmethod
    def __response_dict(response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return deepcopy(response)
        if hasattr(response, "model_dump"):
            return cast(Dict[str, Any], response.model_dump(mode="json"))
        raise ValueError("The Responses API returned an invalid response object.")

    async def __stream_response(
        self,
        stream: AsyncIterator[Any],
        span: Optional[Span],
        legacy_tool_names: Set[str],
        modern_tool_names: Set[str],
    ) -> AsyncIterator[ChatResponse]:
        content = ""
        last_emitted_content: Optional[str] = None
        reasoning_summary = ""
        try:
            async for event in stream:
                event_data = self.__response_dict(event)
                event_type = event_data.get("type")
                if event_type in (
                    "response.output_text.delta",
                    "response.refusal.delta",
                ):
                    delta = event_data.get("delta")
                    if not isinstance(delta, str):
                        raise ValueError(
                            f"Responses stream event {event_type} requires a delta."
                        )
                    content += delta
                    if content != last_emitted_content:
                        last_emitted_content = content
                        yield ChatResponse(
                            error=None,
                            chat_message=ChatMessage(
                                sender=ChatMessageSender.BOT,
                                content=content,
                            ),
                        )
                    continue

                if event_type == "response.reasoning_summary_text.delta":
                    delta = event_data.get("delta")
                    if isinstance(delta, str):
                        reasoning_summary += delta
                    continue

                if event_type == "response.completed":
                    response_data = self.__response_dict(event_data.get("response"))
                    terminal_response = self.__completed_response(
                        response_data=response_data,
                        span=span,
                        legacy_tool_names=legacy_tool_names,
                        modern_tool_names=modern_tool_names,
                        streamed_reasoning_summary=reasoning_summary or None,
                    )
                    terminal_message = terminal_response.chat_message
                    if terminal_message and (
                        terminal_message.content != last_emitted_content
                        or terminal_message.response_replay_items is not None
                        or terminal_message.function_call_request is not None
                        or terminal_message.tool_call_requests is not None
                        or terminal_message.reasoning_summary is not None
                    ):
                        yield terminal_response
                    return

                if event_type in ("response.failed", "response.incomplete"):
                    response_data = self.__response_dict(event_data.get("response"))
                    details = (
                        response_data.get("error")
                        or response_data.get("incomplete_details")
                        or response_data.get("status")
                    )
                    raise ValueError(f"Responses stream did not complete: {details}")

                if event_type == "error":
                    raise ValueError(
                        "Responses stream failed: "
                        f"{event_data.get('message') or event_data.get('code')}"
                    )
            raise ValueError("Responses stream ended before a terminal event.")
        except Exception as error:
            yield self.__error_response(error, span)

    def __completed_response(
        self,
        response_data: Dict[str, Any],
        span: Optional[Span],
        legacy_tool_names: Set[str],
        modern_tool_names: Set[str],
        streamed_reasoning_summary: Optional[str] = None,
    ) -> ChatResponse:
        chat_message = self.__chat_message_from(
            response_data=response_data,
            legacy_tool_names=legacy_tool_names,
            modern_tool_names=modern_tool_names,
        )
        if not chat_message.reasoning_summary and streamed_reasoning_summary:
            chat_message.reasoning_summary = streamed_reasoning_summary
        usage = response_data.get("usage")
        end_span(
            usage=(
                {
                    "usage": {
                        "input": usage.get("input_tokens", 0),
                        "output": usage.get("output_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                        "unit": "TOKENS",
                    }
                }
                if usage
                else {}
            ),
            tool_calls=chat_message.tool_call_requests,
            span=span,
            content=chat_message.content,
            role="assistant",
            function_call=chat_message.function_call_request,
            reasoning_summary=chat_message.reasoning_summary,
        )
        return ChatResponse(error=None, chat_message=chat_message)

    def __error_response(self, error: Exception, span: Optional[Span]) -> ChatResponse:
        if isinstance(error, BadRequestError):
            self.__end_error_span(span, error.message)
            if error.status_code == 400 and (
                "context_length_exceeded" in error.message
                or "string too long" in error.message
            ):
                return ChatResponse(
                    error=generate_prompt_too_long_error(error.message),
                    chat_message=None,
                )
            return ChatResponse(error=error, chat_message=None)

        if isinstance(error, RateLimitError):
            logger.warning(f"Rate limit reached: {error}")
            self.__end_error_span(span, error.message, level="WARNING")
            retry_after = (
                error.response.headers.get("retry-after")
                or error.response.headers.get("x-ratelimit-reset-requests")
                or error.response.headers.get("x-ratelimit-reset-tokens")
            )
            if retry_after is None:
                retry_after_match = re.search(
                    r"retry after (\d+\s*\w+)", error.message, re.IGNORECASE
                )
                retry_after = (
                    retry_after_match.group(1) if retry_after_match else "a moment"
                )
            else:
                retry_after = f"{retry_after} seconds"
            return ChatResponse(
                error=None,
                chat_message=ChatMessage(
                    content=(
                        "You have reached the token rate limit for this model. "
                        f"Please retry after {retry_after}. If this issue persists, "
                        "contact support@zeta-alpha.com for assistance."
                    ),
                    sender=ChatMessageSender.BOT,
                ),
            )

        self.__end_error_span(span, str(error))
        return ChatResponse(error=error, chat_message=None)

    def __chat_message_from(  # noqa: C901
        self,
        response_data: Dict[str, Any],
        legacy_tool_names: Set[str],
        modern_tool_names: Set[str],
    ) -> ChatMessage:
        if response_data.get("status") != "completed":
            details = (
                response_data.get("error")
                or response_data.get("incomplete_details")
                or response_data.get("status")
            )
            raise ValueError(f"Responses API request did not complete: {details}")
        output = response_data.get("output")
        if not isinstance(output, list):
            raise ValueError("Responses API output must be a list.")

        reasoning_summaries = [
            part.get("text", "")
            for item in output
            if isinstance(item, dict) and item.get("type") == "reasoning"
            for part in (item.get("summary") or [])
            if isinstance(part, dict) and part.get("text")
        ]
        reasoning_summary = "\n".join(reasoning_summaries) or None

        content: List[str] = []
        legacy_calls: List[FunctionCallRequest] = []
        modern_calls: List[ToolCallRequest] = []
        for item in output:
            if not isinstance(item, dict):
                raise ValueError("Responses API output items must be objects.")
            item_type = item.get("type")
            if item_type not in ("reasoning", "message", "function_call"):
                raise ValueError(f"Unsupported Responses output item: {item_type}")
            if item.get("status") not in (None, "completed"):
                raise ValueError(f"Responses output item did not complete: {item_type}")
            if item_type == "reasoning":
                continue
            if item_type == "message":
                item_content = item.get("content")
                if not isinstance(item_content, list):
                    raise ValueError("Responses message content must be a list.")
                for part in item_content:
                    if not isinstance(part, dict):
                        raise ValueError("Responses message parts must be objects.")
                    if part.get("type") == "output_text":
                        content.append(cast(str, part.get("text", "")))
                    elif part.get("type") == "refusal":
                        content.append(cast(str, part.get("refusal", "")))
                    else:
                        raise ValueError(
                            "Unsupported Responses message content: "
                            f"{part.get('type')}"
                        )
                continue

            call_id = item.get("call_id")
            name = item.get("name")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Responses function calls require a call_id.")
            if not isinstance(name, str) or not name:
                raise ValueError("Responses function calls require a name.")
            try:
                arguments = json.loads(item.get("arguments", ""))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Function call arguments for {name} are not valid JSON."
                ) from error
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"Function call arguments for {name} must be a JSON object."
                )
            function_call = FunctionCallRequest(
                function_name=name, function_params=arguments
            )
            if name in legacy_tool_names:
                legacy_calls.append(function_call)
            elif name in modern_tool_names:
                modern_calls.append(
                    ToolCallRequest(
                        id=call_id,
                        function_call_request=function_call,
                    )
                )
            else:
                raise ValueError(f"Responses API called unknown function: {name}")

        if legacy_calls and modern_calls:
            raise ValueError("A response cannot mix legacy functions and modern tools.")
        if len(legacy_calls) > 1:
            raise ValueError("Legacy functions support one function call per response.")
        has_function_calls = bool(legacy_calls or modern_calls)
        return ChatMessage(
            sender=ChatMessageSender.BOT,
            content="".join(content),
            function_call_request=(legacy_calls[0] if legacy_calls else None),
            tool_call_requests=modern_calls or None,
            response_replay_items=(deepcopy(output) if has_function_calls else None),
            reasoning_summary=reasoning_summary,
        )

    @staticmethod
    def __trace_messages(conversation: BotConversation) -> List[Dict[str, Any]]:
        messages = [
            message.model_dump(mode="json", exclude={"response_replay_items"})
            for message in conversation.messages
        ]
        if conversation.bot_setup_description is not None:
            messages.insert(
                0,
                {
                    "sender": ChatMessageSender.SYSTEM.value,
                    "content": conversation.bot_setup_description,
                },
            )
        return messages

    @staticmethod
    def __end_error_span(
        span: Optional[Span], status_message: str, level: str = "ERROR"
    ) -> None:
        if span:
            span.end(
                attributes={
                    "level": level,
                    "status_message": status_message,
                }
            )

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: OpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "OpenAiResponsesClient":
        client = openai.AsyncOpenAI(
            api_key=vendor_configuration.openai_api_key.get_unencrypted_secret(),
            organization=vendor_configuration.openai_org.get_unencrypted_secret(),
            base_url=vendor_configuration.openai_api_base,
        )
        return cls(client=client, model_configuration=model_configuration, span=span)


@PromptClientFactory.register(LLMProviderName.OPENAI, LLMModelType.RESPONSES)
class OpenAiResponsesClient2PromptClientAdapter(OpenAiChatClient2PromptClientAdapter):
    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: OpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "OpenAiResponsesClient2PromptClientAdapter":
        chat_client = OpenAiResponsesClient.from_configuration(
            vendor_configuration, model_configuration, span=span
        )
        return cls(chat_client=chat_client)
