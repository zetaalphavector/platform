import asyncio
import json
import re
from typing import Optional, Tuple

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.attempt_lineage import Score
from zav.agents_sdk.adapters.agent_state.rubric import Rubric
from zav.agents_sdk.adapters.tools.sources.self_improvement.attempts.scorer import (
    ScoreContext,
    Scorer,
)
from zav.agents_sdk.domain.agent_creator import AgentCreator
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage

_MAX_VALUE = 10.0

_JUDGE_SYSTEM_PROMPT = """\
You grade a candidate answer against a reference answer. You are the only \
judge; be strict and consistent.

Score 0-10:
- 10: equivalent to the reference in every claim that matters.
- 7-9: correct on the substance, minor omissions or imprecision.
- 4-6: partially correct, at least one material error or gap.
- 1-3: mostly wrong or off-target.
- 0: empty, refuses, or unrelated.

The candidate's author must never learn the reference. In `reasons`, describe \
what is missing or wrong in the candidate in general terms. Never quote, \
paraphrase, or hint at the reference's content or wording.

Reply with JSON only: {"score": <integer 0-10>, "reasons": "<one or two sentences>"}\
"""

_JUDGE_USER_TEMPLATE = """\
{task_section}{rubric_section}<reference_answer>
{reference}
</reference_answer>

<candidate_answer>
{candidate}
</candidate_answer>\
"""


class LLMJudgeScorerConfiguration(BaseModel):
    enabled: bool = Field(
        False, description="Grade against the matched rubric's reference with an LLM."
    )
    judge_agent_identifier: Optional[str] = Field(
        None,
        description=(
            "Agent that grades, run once per judgement in a fresh context with the "
            "grading instructions as its system prompt; its setup picks the model. "
            "Default: a fresh copy of the current agent."
        ),
    )
    repeats: int = Field(
        3, description="Independent judgements to average; also yields the spread."
    )
    pass_threshold: float = Field(
        7.0,
        description=(
            "Mean score (0-10) at or above which an attempt passes, unless the "
            "rubric sets its own."
        ),
    )


def parse_judgement(text: str) -> Tuple[float, str]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge returned no JSON object: {text[:200]!r}")
    payload = json.loads(match.group(0))
    value = float(payload["score"])
    value = max(0.0, min(_MAX_VALUE, value))
    return value, str(payload.get("reasons", "")).strip()


class LLMJudgeScorer(Scorer):
    """The fitness function for free-text tasks. Each judgement is a separate,
    ephemeral agent run (the same mechanism as the reviewer interjection) that
    sees the rubric's reference and returns a score; the reference never enters
    the graded agent's transcript."""

    source_name = "llm_judge"

    def __init__(
        self,
        agent_creator: AgentCreator,
        enabled: bool,
        judge_agent_identifier: Optional[str],
        repeats: int,
        pass_threshold: float,
    ) -> None:
        self.enabled = enabled
        self.__agent_creator = agent_creator
        self.__judge_agent_identifier = judge_agent_identifier
        self.__repeats = max(1, repeats)
        self.__pass_threshold = pass_threshold

    async def score(self, artifact: str, context: ScoreContext) -> Score:
        rubric = context.rubric
        if not rubric.reference.strip():
            return Score(
                scorer=self.source_name,
                value=0.0,
                max_value=_MAX_VALUE,
                passed=False,
                details="The rubric has no reference answer.",
            )
        threshold = (
            rubric.pass_threshold
            if rubric.pass_threshold is not None
            else self.__pass_threshold
        )
        judgements = await asyncio.gather(
            *(self.__judge_once(artifact, rubric) for _ in range(self.__repeats))
        )
        values = [value for value, _ in judgements]
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        reasons = next((r for _, r in reversed(judgements) if r), "")
        return Score(
            scorer=self.source_name,
            value=round(mean, 2),
            max_value=_MAX_VALUE,
            passed=mean >= threshold,
            details=(
                f"mean {mean:.1f}/10 over {len(values)} judgement(s), "
                f"spread {spread:.1f}, threshold {threshold:g}."
            ),
            reasons=reasons or None,
        )

    async def __judge_once(self, artifact: str, rubric: Rubric) -> Tuple[float, str]:
        prompt = _JUDGE_USER_TEMPLATE.format(
            task_section=f"<task>\n{rubric.task}\n</task>\n\n" if rubric.task else "",
            rubric_section=(
                f"<grading_criteria>\n{rubric.criteria}\n</grading_criteria>\n\n"
                if rubric.criteria
                else ""
            ),
            reference=rubric.reference,
            candidate=artifact,
        )
        judge = await self.__agent_creator.create(
            agent_identifier=self.__judge_agent_identifier,
            system_prompt=_JUDGE_SYSTEM_PROMPT,
        )
        verdict = await judge.execute([ChatMessage.as_user(prompt)])
        return parse_judgement(verdict.content if verdict else "")


class LLMJudgeScorerFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        agent_creator: AgentCreator,
        llm_judge_scorer_configuration: LLMJudgeScorerConfiguration = (
            LLMJudgeScorerConfiguration()
        ),
    ) -> LLMJudgeScorer:
        configuration = llm_judge_scorer_configuration
        return LLMJudgeScorer(
            agent_creator=agent_creator,
            enabled=configuration.enabled,
            judge_agent_identifier=configuration.judge_agent_identifier,
            repeats=configuration.repeats,
            pass_threshold=configuration.pass_threshold,
        )
