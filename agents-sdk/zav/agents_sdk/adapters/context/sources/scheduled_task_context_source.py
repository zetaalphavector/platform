import json
from typing import Any, Dict, List, Literal, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ResolvedContextItem,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import (
    ChatMessage,
    ChatMessageSender,
    ConversationContext,
)

_SCHEDULED_TASK_INSTRUCTIONS = (
    "Rely not only on the instructions of the scheduled task, but also on the "
    "previous messages in the conversation which may contain additional "
    "information about how to execute the task. Take into account the scheduling "
    "configuration as well and provide an answer focusing on the new information "
    "found after the last execution of the scheduled task, if any. If there are "
    "no new updated information since the last execution, inform the user, and "
    "do not repeat the same information. If there are new information, you "
    "should support each fact you state in your answer with inline citations "
    "to the provided search results."
)


class ScheduledTaskContextSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False, description="Enable scheduled task context resolution."
    )


class ScheduledTaskContextSource(ContextSource):

    source_name = "scheduled_task"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.__task_params: Optional[Dict[str, Any]] = None

    def describe(self, origin: ContextOrigin) -> str:
        return "This is a scheduled task execution. " + _SCHEDULED_TASK_INSTRUCTIONS

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        return None

    async def resolve_from_conversation(
        self,
        conversation: List[ChatMessage],
    ) -> Optional[List[ResolvedContextItem]]:
        for message in reversed(conversation):
            if message.sender != ChatMessageSender.USER:
                continue
            if not message.content_parts:
                continue
            for part in message.content_parts:
                if not part.tool:
                    continue
                if part.tool.name == "execute_scheduled_task":
                    self.__task_params = part.tool.params or {}
                    return [
                        ResolvedContextItem(
                            id="scheduled_task",
                            data=self.__task_params,
                            item_type="scheduled_task",
                        )
                    ]
        return None

    def format_items(
        self,
        items: List[ResolvedContextItem],
        content_format: Literal["json", "xml"],
        max_item_content_length: Optional[int] = None,
    ) -> str:
        if not self.__task_params:
            return ""
        return json.dumps(self.__task_params, indent=2, default=str)


class ScheduledTaskContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        scheduled_task_context_source_configuration: ScheduledTaskContextSourceConfiguration = ScheduledTaskContextSourceConfiguration(),  # noqa: E501
    ) -> ScheduledTaskContextSource:
        return ScheduledTaskContextSource(
            enabled=scheduled_task_context_source_configuration.enabled,
        )
