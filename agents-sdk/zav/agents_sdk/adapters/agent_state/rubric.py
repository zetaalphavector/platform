import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.chat_message import ChatMessage, ChatMessageSender

REGEX_PREFIX = "re:"


class Rubric(BaseModel):
    """One task the agent may be asked and how a candidate answer is graded.
    Matching is deterministic: `match` is a case-insensitive substring of the
    user's latest message, or a regular expression when prefixed with `re:`."""

    name: str = Field(..., description="Task name; keys the attempt lineage.")
    match: str = Field(
        ..., description="Substring of the user message, or `re:<pattern>`."
    )
    reference: str = Field(
        ..., description="The gold answer. Never shown to the agent."
    )
    task: Optional[str] = Field(
        None, description="What the question asks, for the grader's context."
    )
    criteria: Optional[str] = Field(
        None, description="Extra grading criteria for the grader."
    )
    pass_threshold: Optional[float] = Field(
        None, description="Overrides the scorer's default pass threshold."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured data a scorer or harness needs beyond the reference text, "
            "such as the option list of a multiple-choice question or the prompt "
            "that poses the task."
        ),
    )

    def matches(self, text: str) -> bool:
        if self.match.startswith(REGEX_PREFIX):
            return (
                re.search(self.match[len(REGEX_PREFIX) :], text, re.DOTALL) is not None
            )
        return self.match.lower() in text.lower()


class RubricSource(ABC):
    source_name: ClassVar[str]
    enabled: bool

    @abstractmethod
    async def list_rubrics(self) -> List[Rubric]:
        raise NotImplementedError


class RubricSourceGroup(DependencyGroup[RubricSource]):
    __collects__ = RubricSource


def latest_user_message(conversation: List[ChatMessage]) -> str:
    for message in reversed(conversation):
        if message.sender == ChatMessageSender.USER:
            return message.content or ""
    return ""


def match_rubric(
    rubrics: List[Rubric], conversation: List[ChatMessage]
) -> Optional[Rubric]:
    text = latest_user_message(conversation)
    if not text:
        return None
    for rubric in rubrics:
        if rubric.matches(text):
            return rubric
    return None
