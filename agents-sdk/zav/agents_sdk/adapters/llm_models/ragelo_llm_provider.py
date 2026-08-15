import json
from typing import List, Optional, Tuple, TypeVar

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
from zav.pydantic_compat import BaseModel, ValidationError

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
        max_retries: int = 2,
    ) -> None:
        super().__init__(config=None)
        self.__chat_completion_client = chat_completion_client
        self.__max_tokens = max_tokens
        self.__max_retries = max_retries

    async def call_async(
        self,
        input: LLMInputPrompt,
        response_schema: type[T_Schema],
    ) -> LLMResponseType[T_Schema]:
        messages = self.__build_messages(input)
        tool_name = response_schema.__name__
        tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    response_schema.__doc__
                    or "Return the answer as arguments to this tool."
                ),
                "parameters": response_schema.model_json_schema(),
            },
        }

        last_error: Exception | None = None
        for _ in range(self.__max_retries + 1):
            request: ChatClientRequest = {
                "conversation": BotConversation(messages=messages),
                "max_tokens": self.__max_tokens,
                "tools": [tool],
                "tool_choice": tool_name,
            }

            response = await self.__chat_completion_client.complete(
                request, stream=False
            )

            if response.error:
                last_error = ValueError(f"LLM request failed: {response.error}")
                continue
            if not response.chat_message:
                last_error = ValueError("LLM returned an empty response")
                continue

            raw_answer, params = self.__extract_answer(response.chat_message)
            if raw_answer is None:
                last_error = ValueError("LLM returned an empty response")
                continue

            try:
                if params is not None:
                    parsed_answer = response_schema.model_validate(params)
                else:
                    parsed_answer = response_schema.model_validate_json(raw_answer)
            except ValidationError as e:
                last_error = e
                messages = messages + [
                    PcChatMessage(sender=ChatMessageSender.BOT, content=raw_answer),
                    PcChatMessage(
                        sender=ChatMessageSender.USER,
                        content=(
                            "The previous response did not match the required schema. "
                            f"Validation error:\n{e}\n"
                            f"Correct it and call the {tool_name} tool again."
                        ),
                    ),
                ]
                continue

            return LLMResponseType(
                raw_answer=raw_answer,
                parsed_answer=parsed_answer,
            )

        raise ValueError(
            f"Failed to get a valid {response_schema.__name__} response from the LLM "
            f"after {self.__max_retries + 1} attempts: {last_error}"
        )

    def __build_messages(self, input: LLMInputPrompt) -> List[PcChatMessage]:
        messages: List[PcChatMessage] = []
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
        return messages

    @staticmethod
    def __extract_answer(
        chat_message: PcChatMessage,
    ) -> Tuple[Optional[str], Optional[dict]]:
        if chat_message.tool_call_requests:
            params = (
                chat_message.tool_call_requests[0].function_call_request.function_params
                or {}
            )
            return json.dumps(params), params
        if chat_message.content:
            return chat_message.content, None
        return None, None
