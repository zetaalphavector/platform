import json
from typing import TypeVar

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from zav.prompt_completion import (
    BotConversation,
    ChatClientRequest,
    ChatCompletionClient,
)
from zav.prompt_completion import ChatMessage as PcChatMessage
from zav.prompt_completion import (
    ChatMessageSender,
)
from zav.pydantic_compat import BaseModel

T_Schema = TypeVar("T_Schema", bound=BaseModel)

ROLE_TO_SENDER = {
    "system": ChatMessageSender.SYSTEM,
    "user": ChatMessageSender.USER,
    "assistant": ChatMessageSender.BOT,
}


class ZAVRageloLLMProvider(BaseLLMProvider):

    def __init__(
        self,
        chat_completion_client: ChatCompletionClient,
        max_tokens: int = 2048,
    ) -> None:
        super().__init__(config=None)
        self.__chat_completion_client = chat_completion_client
        self.__max_tokens = max_tokens

    async def call_async(
        self,
        input: LLMInputPrompt,
        response_schema: type[T_Schema],
    ) -> LLMResponseType[T_Schema]:
        messages: list[PcChatMessage] = []

        if input.messages:
            for msg in input.messages:
                sender = ROLE_TO_SENDER.get(msg["role"], ChatMessageSender.USER)
                messages.append(PcChatMessage(sender=sender, content=msg["content"]))
        else:
            if input.system_prompt:
                messages.append(
                    PcChatMessage(
                        sender=ChatMessageSender.SYSTEM,
                        content=input.system_prompt,
                    )
                )
            if input.user_message:
                messages.append(
                    PcChatMessage(
                        sender=ChatMessageSender.USER,
                        content=input.user_message,
                    )
                )

        if not messages:
            raise ValueError("No input provided")

        schema = json.dumps(response_schema.model_json_schema(), indent=4)
        schema_instruction = (
            "\n\nYour output should be a JSON string that STRICTLY "
            f"adheres to the following schema:\n{schema}"
        )
        last_user_msg = None
        for msg in reversed(messages):
            if msg.sender == ChatMessageSender.USER:
                last_user_msg = msg
                break
        if last_user_msg:
            last_user_msg.content += schema_instruction
        else:
            messages.append(
                PcChatMessage(
                    sender=ChatMessageSender.USER,
                    content=schema_instruction.strip(),
                )
            )

        request: ChatClientRequest = {
            "conversation": BotConversation(messages=messages),
            "max_tokens": self.__max_tokens,
        }

        response = await self.__chat_completion_client.complete(request, stream=False)

        if response.error:
            raise ValueError(f"LLM request failed: {response.error}")
        if not response.chat_message or not response.chat_message.content:
            raise ValueError("LLM returned an empty response")

        raw_answer = response.chat_message.content
        parsed_answer = response_schema.model_validate_json(raw_answer)

        return LLMResponseType(
            raw_answer=raw_answer,
            parsed_answer=parsed_answer,
        )
