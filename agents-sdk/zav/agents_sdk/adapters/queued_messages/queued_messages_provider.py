from typing import List, Optional, Set

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion
from zav.agents_sdk.adapters.llm_models.queued_messages import QueuedMessages
from zav.agents_sdk.adapters.queued_messages.queued_messages_source import (
    QueuedMessagesSource,
    QueuedMessagesSourceGroup,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage
from zav.agents_sdk.domain.turn_inbox import TurnInbox


class QueuedMessagesProviderConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Read the messages the user sends while a turn runs before the "
            "model's next call. Off, they wait for the next turn."
        ),
    )
    include_sources: Optional[List[str]] = Field(
        None, description="Allowlist of queued messages source names to include."
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of queued messages source names to exclude."
    )


class QueuedMessagesProvider(QueuedMessages):
    """Reads the interactive turn's inbox for the root agent and gives the
    model what its active sources make of each message. Off without an inbox
    (no interactive turn) or without an active source."""

    def __init__(
        self,
        sources: List[QueuedMessagesSource],
        enabled: bool,
        include_sources: Optional[Set[str]],
        exclude_sources: Optional[Set[str]],
        inbox: Optional[TurnInbox],
    ) -> None:
        self.__sources = sources
        self.__enabled = enabled
        self.__include_sources = include_sources
        self.__exclude_sources = exclude_sources
        self.__inbox = inbox
        self.__cached_active: Optional[List[QueuedMessagesSource]] = None

    @property
    def enabled(self) -> bool:
        return self.__inbox is not None and bool(self.__active_sources())

    async def take(self) -> List[ChatMessage]:
        return await self.__inbox.messages() if self.__inbox is not None else []

    async def has_messages(self) -> bool:
        return self.__inbox is not None and await self.__inbox.has_messages()

    async def frame(self, messages: List[ChatMessage]) -> List[ChatCompletion]:
        completions: List[ChatCompletion] = []
        for source in self.__active_sources():
            completions.extend(await source.frame(messages))
        return completions

    def __active_sources(self) -> List[QueuedMessagesSource]:
        if self.__cached_active is not None:
            return self.__cached_active
        self.__cached_active = (
            [
                source
                for source in self.__sources
                if is_source_active(
                    source.source_name,
                    source.enabled,
                    self.__include_sources,
                    self.__exclude_sources,
                )
            ]
            if self.__enabled
            else []
        )
        return self.__cached_active


class QueuedMessagesProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        queued_messages_source_group: QueuedMessagesSourceGroup = (
            QueuedMessagesSourceGroup(items=[])
        ),
        queued_messages_provider_configuration: QueuedMessagesProviderConfiguration = (
            QueuedMessagesProviderConfiguration()
        ),
        turn_inbox: Optional[TurnInbox] = None,
    ) -> QueuedMessages:
        return QueuedMessagesProvider(
            sources=queued_messages_source_group.items,
            enabled=queued_messages_provider_configuration.enabled,
            include_sources=(
                set(queued_messages_provider_configuration.include_sources)
                if queued_messages_provider_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(queued_messages_provider_configuration.exclude_sources)
                if queued_messages_provider_configuration.exclude_sources
                else None
            ),
            inbox=turn_inbox,
        )
