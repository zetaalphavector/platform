from typing import List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.attempt_lineage import AttemptLineage
from zav.agents_sdk.adapters.agent_state.rubric import (
    Rubric,
    RubricSource,
    RubricSourceGroup,
    match_rubric,
)
from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ResolvedContextItem,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext


class RubricContextSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Match the latest user message against the rubric sources and select "
            "the task for the attempts gate. Nothing is added to the prompt."
        ),
    )


class RubricContextSource(ContextSource):
    """What the conversation implies about the turn: whether it is a known task.
    Matches the latest user message against the agent's rubrics while the
    context provider processes the conversation, and selects the winner into
    the shared ``AttemptLineage`` so the attempts provider and the attempt
    interjection sources engage. Contributes no context of its own; the rubric
    and its reference never reach the prompt."""

    source_name = "rubric"
    source_description = "Whether the request matches a known task and its rubric."
    source_gate = "attempt_lineage"

    def __init__(
        self,
        lineage: AttemptLineage,
        rubric_sources: List[RubricSource],
        enabled: bool,
    ) -> None:
        self.enabled = enabled
        self.__lineage = lineage
        self.__rubric_sources = rubric_sources

    def describe(self, origin: ContextOrigin) -> str:
        return ""

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        return None

    async def resolve_from_conversation(
        self, conversation: List[ChatMessage]
    ) -> Optional[List[ResolvedContextItem]]:
        if not self.__lineage.enabled:
            return None
        rubrics: List[Rubric] = []
        for source in self.__rubric_sources:
            if source.enabled:
                rubrics.extend(await source.list_rubrics())
        self.__lineage.select(match_rubric(rubrics, conversation))
        return None


class RubricContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        attempt_lineage: AttemptLineage,
        rubric_source_group: RubricSourceGroup = RubricSourceGroup(items=[]),
        rubric_context_source_configuration: RubricContextSourceConfiguration = (
            RubricContextSourceConfiguration()
        ),
    ) -> RubricContextSource:
        return RubricContextSource(
            lineage=attempt_lineage,
            rubric_sources=rubric_source_group.items,
            enabled=rubric_context_source_configuration.enabled,
        )
