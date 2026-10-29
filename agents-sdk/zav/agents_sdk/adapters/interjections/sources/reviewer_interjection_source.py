from typing import Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.interjections.interjection_source import (
    InterjectionSource,
)
from zav.agents_sdk.adapters.interjections.sources.repeated_tool_failure_interjection_source import (  # noqa: E501
    trailing_failure_streak,
)
from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletionSender,
)
from zav.agents_sdk.adapters.llm_models.chat_turn_request import InnerLoopContext
from zav.agents_sdk.adapters.llm_models.transcript_rendering import (
    render_transcript,
)
from zav.agents_sdk.domain.agent_creator import AgentCreator, run_sub_agent
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_DEFAULT_INSTRUCTIONS = """\
You are reviewing another agent's work in progress. Below is its transcript so \
far: the user's request, its reasoning, the tools it called and what came back. \
Its progress has stalled.

Diagnose in one or two sentences why it is stuck, then propose two or three \
concrete, different directions it should try next. Do not solve the task \
yourself and do not repeat directions the transcript shows were already tried. \
Keep the whole reply under 150 words.\
"""


class ReviewerInterjectionSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False, description="Have a fresh sub-agent review the transcript on a stall."
    )
    on_consecutive_tool_failures: Optional[int] = Field(
        3,
        description=(
            "Trigger when one tool has failed this many times in a row. None "
            "disables this trigger."
        ),
    )
    every_n_tool_steps: Optional[int] = Field(
        None,
        description=(
            "Also trigger every N tool steps regardless of failures. None disables."
        ),
    )
    reviewer_agent_identifier: Optional[str] = Field(
        None,
        description=(
            "Agent to run the review with. Defaults to a fresh copy of the current "
            "agent."
        ),
    )
    max_transcript_chars: int = Field(
        24_000, description="Tail of the rendered transcript handed to the reviewer."
    )
    instructions: Optional[str] = Field(
        None, description="Override the reviewer instructions."
    )


class ReviewerInterjectionSource(InterjectionSource):
    """The supervisor: a separate agent with a clean context reads the trajectory
    and steers, instead of the stuck agent reviewing itself."""

    source_name = "reviewer"
    source_description = "A separate reviewer reads the agent's progress and steers it."

    def __init__(
        self,
        agent_creator: AgentCreator,
        enabled: bool,
        on_consecutive_tool_failures: Optional[int],
        every_n_tool_steps: Optional[int],
        reviewer_agent_identifier: Optional[str],
        max_transcript_chars: int,
        instructions: Optional[str],
    ) -> None:
        self.enabled = enabled
        self.__agent_creator = agent_creator
        self.__on_consecutive_tool_failures = on_consecutive_tool_failures
        self.__every_n_tool_steps = every_n_tool_steps
        self.__reviewer_agent_identifier = reviewer_agent_identifier
        self.__max_transcript_chars = max_transcript_chars
        self.__instructions = instructions or _DEFAULT_INSTRUCTIONS
        self.__last_failure_marker: Optional[tuple] = None
        self.__last_step_marker: Optional[int] = None

    async def before_model_call(self, context: InnerLoopContext) -> Optional[str]:
        if not self.__should_review(context):
            return None
        prompt = "\n\n".join(
            [
                self.__instructions,
                "<transcript>",
                render_transcript(context.transcript, self.__max_transcript_chars),
                "</transcript>",
            ]
        )
        try:
            review = await run_sub_agent(
                agent_creator=self.__agent_creator,
                prompt=prompt,
                target_identifier=self.__reviewer_agent_identifier,
            )
        except Exception as e:
            logger.error(f"Reviewer interjection failed: {e}")
            return None
        return review.content.strip() or None

    def __should_review(self, context: InnerLoopContext) -> bool:
        triggered = False
        threshold = self.__on_consecutive_tool_failures
        if threshold:
            streak = trailing_failure_streak(context.transcript)
            if streak is not None and streak[1] >= threshold:
                marker = (streak[0], streak[1])
                if streak[1] % threshold == 0 and marker != self.__last_failure_marker:
                    self.__last_failure_marker = marker
                    triggered = True
        every = self.__every_n_tool_steps
        if every:
            steps = sum(
                1 for c in context.transcript if c.sender == ChatCompletionSender.TOOL
            )
            if steps and steps % every == 0 and steps != self.__last_step_marker:
                self.__last_step_marker = steps
                triggered = True
        return triggered


class ReviewerInterjectionSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        agent_creator: AgentCreator,
        reviewer_interjection_source_configuration: (
            ReviewerInterjectionSourceConfiguration
        ) = ReviewerInterjectionSourceConfiguration(),
    ) -> ReviewerInterjectionSource:
        configuration = reviewer_interjection_source_configuration
        return ReviewerInterjectionSource(
            agent_creator=agent_creator,
            enabled=configuration.enabled,
            on_consecutive_tool_failures=configuration.on_consecutive_tool_failures,
            every_n_tool_steps=configuration.every_n_tool_steps,
            reviewer_agent_identifier=configuration.reviewer_agent_identifier,
            max_transcript_chars=configuration.max_transcript_chars,
            instructions=configuration.instructions,
        )
