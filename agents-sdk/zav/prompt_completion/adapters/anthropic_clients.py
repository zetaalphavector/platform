import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union, overload

import anthropic
from anthropic.types import Message as AnthropicMessage
from typing_extensions import Literal
from zav.llm_domain import (
    AnthropicConfiguration,
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
    PromptTooLargeError,
)
from zav.prompt_completion.client_factories import ChatClientFactory


def build_client(vendor_configuration: AnthropicConfiguration):
    if vendor_configuration.anthropic_api_type == "bedrock":
        return anthropic.AsyncAnthropicBedrock(
            # The Bedrock client uses the AWS_SECRET_ACCESS_KEY & AWS_ACCESS_KEY_ID
            # environment variables for authentication, and the AWS_REGION variable
            # to determine the region. We can override these values by passing them
            # explicitly here to allow per-tenant configuration.
            aws_secret_key=(
                vendor_configuration.aws_secret_key.get_unencrypted_secret()
                if vendor_configuration.aws_secret_key
                else None
            ),
            aws_access_key=(
                vendor_configuration.aws_access_key.get_unencrypted_secret()
                if vendor_configuration.aws_access_key
                else None
            ),
            aws_region=vendor_configuration.aws_region,
            # This is optional, if unset it will use either the value from the
            # "ANTHROPIC_BEDROCK_BASE_URL" environment variable or the default
            # Bedrock URL: https://bedrock-runtime.{region}.amazonaws.com
            base_url=vendor_configuration.anthropic_api_base,
        )
    else:
        return anthropic.AsyncAnthropic(
            api_key=vendor_configuration.anthropic_api_key.get_unencrypted_secret(),
            base_url=vendor_configuration.anthropic_api_base,
        )


@ChatClientFactory.register(LLMProviderName.ANTHROPIC, LLMModelType.CHAT)
class AnthropicChatClient(ChatCompletionClient):
    __SENDER_TO_ROLE = {
        ChatMessageSender.BOT: "assistant",
        ChatMessageSender.USER: "user",
    }
    __ROLE_TO_SENDER = {
        "assistant": ChatMessageSender.BOT,
        "user": ChatMessageSender.USER,
    }

    def __init__(
        self,
        client: Union[anthropic.AsyncAnthropic, anthropic.AsyncAnthropicBedrock],
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ):
        self.__client = client
        self.__model_name = model_configuration.name
        self.__model_temperature = model_configuration.temperature
        self.__span = span

    def __messages_from(
        self,
        conversation: BotConversation,
    ) -> Tuple[List[Any], Optional[str]]:
        messages: List[Any] = []
        system_prompt: Optional[str] = conversation.bot_setup_description
        for message in conversation.messages:
            content: List[Dict[str, Any]] = []
            if message.image_uri:
                m = re.match(r"data:(.*);base64,(.*)", message.image_uri)
                if not m:
                    # Anthropic only supports base64 images currently
                    raise ValueError(f"Invalid image_uri format: {message.image_uri}")
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": m.group(1),
                            "data": m.group(2),
                        },
                    }
                )
            if message.content:
                content.append({"type": "text", "text": message.content})
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

        return messages, system_prompt

    def __chat_message_from(self, response: AnthropicMessage) -> ChatMessage:
        message = ChatMessage(
            content=response.content[0].text,  # type: ignore
            sender=self.__ROLE_TO_SENDER[response.role],
        )
        return message

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
                "Streaming is not supported yet for Anthropic client."
            )
        try:
            messages, system_prompt = self.__messages_from(request["conversation"])
        except ValueError as e:
            return ChatResponse(error=e, chat_message=None)
        try:
            generation_span = create_span(
                messages=messages,
                model_name=self.__model_name,
                model_temperature=self.__model_temperature,
                span=self.__span,
                max_tokens=request["max_tokens"],
                stream=stream,
            )
            response = await self.__client.messages.create(
                model=self.__model_name,
                messages=messages,
                max_tokens=request["max_tokens"],
                temperature=self.__model_temperature,
                **({"system": system_prompt} if system_prompt else {}),  # type: ignore
            )
            end_span(
                usage=(
                    {
                        "usage": {
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens,
                            "total": response.usage.input_tokens
                            + response.usage.output_tokens,
                            "unit": "TOKENS",
                        }
                    }
                    if response.usage
                    else {}
                ),
                span=generation_span,
                content=response.content[0].text,
                role=response.role,
            )
            chat_message = self.__chat_message_from(response)
            return ChatResponse(error=None, chat_message=chat_message)
        except anthropic.BadRequestError as e:
            if generation_span:
                generation_span.end(
                    attributes={
                        "level": "ERROR",
                        "status_message": e.message,
                    }
                )
            if "prompt is too long" in e.message:
                extra_tokens = None
                if m := re.search(
                    r"prompt is too long: (\d+) tokens > (\d+) maximum", e.message
                ):
                    extra_tokens = int(m.group(1)) - int(m.group(2))
                return ChatResponse(
                    error=PromptTooLargeError(e.message, extra_tokens=extra_tokens),
                    chat_message=None,
                )
            return ChatResponse(error=e, chat_message=None)
        except Exception as error:
            if generation_span:
                generation_span.end(
                    attributes={
                        "level": "ERROR",
                        "status_message": str(error),
                    }
                )
            return ChatResponse(error=error, chat_message=None)

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: AnthropicConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "AnthropicChatClient":
        client = build_client(vendor_configuration)
        return cls(client, model_configuration, span=span)
