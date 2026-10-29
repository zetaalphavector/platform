from typing import List

from zav.pydantic_compat import PYDANTIC_V2, BaseModel, Field

from zav.agents_sdk.adapters.context.context_provider import ContextProvider
from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
)
from zav.agents_sdk.adapters.queued_messages.queued_messages_source import (
    QueuedMessagesSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage

# A message sent mid-turn is guidance for the task in progress, not a new turn:
# without this framing models answer it and end the turn early.
_WORDING = (
    "The user sent a new message while you were working:\n{message}\n\n"
    "Take it into account and continue the task you were doing until it is "
    "done; do not stop after acknowledging it."
)


class MidTurnQueuedMessagesSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Give the model each message sent while the turn runs as guidance "
            "for the task in progress."
        ),
    )
    wording: str = Field(
        _WORDING,
        description=(
            "How the message reaches the model; `{message}` is replaced by its text."
        ),
    )


class MidTurnQueuedMessagesSource(QueuedMessagesSource):
    """Resolves a queued message's attachments through the agent's context
    sources, as for the messages that start a turn, and wraps its text so the
    model continues the task instead of ending the turn."""

    source_name = "mid_turn"
    source_description = "Messages sent mid-turn steer the task in progress."

    def __init__(
        self, context_provider: ContextProvider, enabled: bool, wording: str
    ) -> None:
        self.__context_provider = context_provider
        self.enabled = enabled
        self.__wording = wording

    async def frame(self, messages: List[ChatMessage]) -> List[ChatCompletion]:
        completions = await self.__context_provider.process_conversation(
            initial_context=None, conversation=messages
        )
        return [
            (
                self.__worded(completion)
                if completion.sender == ChatCompletionSender.USER
                else completion
            )
            for completion in completions
        ]

    def __worded(self, completion: ChatCompletion) -> ChatCompletion:
        content = self.__wording.replace("{message}", completion.content or "")
        if PYDANTIC_V2:
            return completion.model_copy(update={"content": content})
        return completion.copy(update={"content": content})


class MidTurnQueuedMessagesSourceFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        context_provider: ContextProvider,
        mid_turn_queued_messages_source_configuration: (
            MidTurnQueuedMessagesSourceConfiguration
        ) = MidTurnQueuedMessagesSourceConfiguration(),
    ) -> MidTurnQueuedMessagesSource:
        configuration = mid_turn_queued_messages_source_configuration
        return MidTurnQueuedMessagesSource(
            context_provider=context_provider,
            enabled=configuration.enabled,
            wording=configuration.wording,
        )
