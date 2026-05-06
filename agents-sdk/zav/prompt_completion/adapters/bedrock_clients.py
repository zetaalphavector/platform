import asyncio
import base64
import re
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import boto3
from typing_extensions import AsyncIterator, Literal
from zav.llm_domain import (
    BedrockConfiguration,
    LLMModelConfiguration,
    LLMModelType,
    LLMProviderName,
)
from zav.llm_tracing import Span

from zav.prompt_completion.adapters.tracing import create_span, end_span
from zav.prompt_completion.client import (
    BotConversation,
    ChatClientRequest,
    ChatCompletionClient,
    ChatMessage,
    ChatMessageSender,
    ChatResponse,
    PromptAnswer,
    PromptCompletionClient,
    PromptResponse,
)
from zav.prompt_completion.client_factories import (
    ChatClientFactory,
    PromptClientFactory,
)

_MIME_TO_BEDROCK_FORMAT = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

_SENDER_TO_ROLE = {
    ChatMessageSender.BOT: "assistant",
    ChatMessageSender.USER: "user",
    ChatMessageSender.FUNCTION: "user",
    ChatMessageSender.TOOL: "user",
    ChatMessageSender.DEVELOPER: "user",
    ChatMessageSender.SYSTEM: "user",
}


def _parse_image(image_uri: str) -> Optional[Dict[str, Any]]:
    m = re.match(r"data:([^;]+);base64,(.+)", image_uri)
    if not m:
        return None
    mime_type = m.group(1)
    image_bytes = base64.b64decode(m.group(2))
    fmt = _MIME_TO_BEDROCK_FORMAT.get(mime_type, "jpeg")
    return {
        "image": {
            "format": fmt,
            "source": {"bytes": image_bytes},
        }
    }


def _chat_message_from_response(response: Dict[str, Any]) -> ChatMessage:
    output = response.get("output", {})
    message = output.get("message", {})
    role = message.get("role", "assistant")
    content_blocks = message.get("content", [])

    text_parts = []
    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])

    return ChatMessage(
        content="".join(text_parts),
        sender=(
            ChatMessageSender.BOT if role == "assistant" else ChatMessageSender.USER
        ),
    )


def _messages_from_conversation(
    conversation: BotConversation,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, str]]]]:
    messages: List[Dict[str, Any]] = []
    system: Optional[List[Dict[str, str]]] = None

    if conversation.bot_setup_description:
        system = [{"text": conversation.bot_setup_description}]

    for message in conversation.messages:
        content: List[Dict[str, Any]] = []

        if message.content_parts:
            for part in message.content_parts:
                if part.text:
                    content.append({"text": part.text})
                if part.image:
                    image_block = _parse_image(part.image.image_uri)
                    if image_block:
                        content.append(image_block)
        elif message.image_uri:
            image_block = _parse_image(message.image_uri)
            if image_block:
                content.append(image_block)
            if message.content:
                content.append({"text": message.content})
        elif message.content:
            content.append({"text": message.content})

        if not content:
            continue

        messages.append(
            {
                "role": _SENDER_TO_ROLE[message.sender],
                "content": content,
            }
        )

    return messages, system


def _build_client(
    vendor_configuration: BedrockConfiguration,
) -> Any:
    kwargs: Dict[str, Any] = {
        "service_name": "bedrock-runtime",
        "region_name": vendor_configuration.aws_region,
    }
    if vendor_configuration.aws_access_key and vendor_configuration.aws_secret_key:
        kwargs["aws_access_key_id"] = (
            vendor_configuration.aws_access_key.get_unencrypted_secret()
        )
        kwargs["aws_secret_access_key"] = (
            vendor_configuration.aws_secret_key.get_unencrypted_secret()
        )
    if vendor_configuration.endpoint_url:
        kwargs["endpoint_url"] = vendor_configuration.endpoint_url
    return boto3.client(**kwargs)


@ChatClientFactory.register(LLMProviderName.BEDROCK, LLMModelType.CHAT)
class BedrockChatClient(ChatCompletionClient):
    def __init__(
        self,
        client: Any,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ):
        self.__client = client
        self.__model_name = model_configuration.name
        self.__model_temperature = model_configuration.temperature
        self.__max_tokens = model_configuration.max_tokens
        self.__span = span

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: BedrockConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "BedrockChatClient":
        client = _build_client(vendor_configuration)
        return cls(
            client=client,
            model_configuration=model_configuration,
            span=span,
        )

    def __converse(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, str]]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        call_kwargs: Dict[str, Any] = {
            "modelId": self.__model_name,
            "messages": messages,
            "inferenceConfig": {
                "temperature": self.__model_temperature,
                "maxTokens": max_tokens,
            },
        }
        if system:
            call_kwargs["system"] = system
        return self.__client.converse(**call_kwargs)

    async def __call_converse(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, str]]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.__converse, messages, system, max_tokens)

    @overload
    async def complete(  # type: ignore
        self, request: ChatClientRequest, stream: Literal[False] = False
    ) -> ChatResponse: ...

    @overload
    async def complete(
        self, request: ChatClientRequest, stream: Literal[True] = True
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
        if stream:
            raise NotImplementedError(
                "Streaming is not supported for the Bedrock Converse client."
            )

        try:
            messages, system = _messages_from_conversation(request["conversation"])
        except ValueError as e:
            return ChatResponse(error=e, chat_message=None)

        max_tokens = request.get("max_tokens") or self.__max_tokens or 4096

        generation_span = create_span(
            messages=messages,
            model_name=self.__model_name,
            model_temperature=self.__model_temperature,
            span=self.__span,
            max_tokens=max_tokens,
            stream=False,
        )

        try:
            response = await self.__call_converse(messages, system, max_tokens)
        except Exception as error:
            if generation_span:
                generation_span.end(
                    attributes={"level": "ERROR", "status_message": str(error)}
                )
            return ChatResponse(error=error, chat_message=None)

        chat_message = _chat_message_from_response(response)

        usage = response.get("usage", {})
        end_span(
            usage=(
                {
                    "usage": {
                        "input": usage.get("inputTokens", 0),
                        "output": usage.get("outputTokens", 0),
                        "total": usage.get("inputTokens", 0)
                        + usage.get("outputTokens", 0),
                        "unit": "TOKENS",
                    }
                }
                if usage
                else {}
            ),
            span=generation_span,
            content=chat_message.content,
            role="assistant",
        )

        return ChatResponse(error=None, chat_message=chat_message)


@PromptClientFactory.register(LLMProviderName.BEDROCK, LLMModelType.CHAT)
class BedrockChatClient2PromptClientAdapter(PromptCompletionClient):
    def __init__(self, chat_client: BedrockChatClient):
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
        vendor_configuration: BedrockConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "BedrockChatClient2PromptClientAdapter":
        chat_client = BedrockChatClient.from_configuration(
            vendor_configuration, model_configuration, span=span
        )
        return cls(chat_client=chat_client)
