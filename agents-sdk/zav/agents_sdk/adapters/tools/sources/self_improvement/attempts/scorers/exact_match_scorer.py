import re

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.attempt_lineage import Score
from zav.agents_sdk.adapters.tools.sources.self_improvement.attempts.scorer import (
    ScoreContext,
    Scorer,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class ExactMatchScorerConfiguration(BaseModel):
    enabled: bool = Field(
        False, description="Pass when the artifact matches the rubric's reference."
    )
    contains: bool = Field(
        False,
        description="Pass when the reference appears anywhere in the artifact.",
    )
    normalize: bool = Field(
        True, description="Casefold and collapse whitespace before comparing."
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


class ExactMatchScorer(Scorer):
    """Deterministic scorer: the rubric's reference is the one accepted answer;
    several alternatives are one per line."""

    source_name = "exact_match"

    def __init__(self, enabled: bool, contains: bool, normalize: bool) -> None:
        self.enabled = enabled
        self.__contains = contains
        self.__normalize = normalize

    async def score(self, artifact: str, context: ScoreContext) -> Score:
        expected = [
            line for line in context.rubric.reference.splitlines() if line.strip()
        ]
        if not expected:
            return Score(
                scorer=self.source_name,
                value=0.0,
                passed=False,
                details="The rubric has no reference answer.",
            )
        candidate = _normalize(artifact) if self.__normalize else artifact
        for answer in expected:
            target = _normalize(answer) if self.__normalize else answer
            matched = target in candidate if self.__contains else candidate == target
            if matched:
                return Score(scorer=self.source_name, value=1.0, passed=True)
        return Score(
            scorer=self.source_name,
            value=0.0,
            passed=False,
            details="The artifact does not match the expected answer.",
        )


class ExactMatchScorerFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        exact_match_scorer_configuration: ExactMatchScorerConfiguration = (
            ExactMatchScorerConfiguration()
        ),
    ) -> ExactMatchScorer:
        return ExactMatchScorer(
            enabled=exact_match_scorer_configuration.enabled,
            contains=exact_match_scorer_configuration.contains,
            normalize=exact_match_scorer_configuration.normalize,
        )
