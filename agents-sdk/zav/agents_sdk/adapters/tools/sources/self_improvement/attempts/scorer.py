from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.agent_state.attempt_lineage import Score
from zav.agents_sdk.adapters.agent_state.rubric import Rubric
from zav.agents_sdk.domain.agent_dependency import DependencyGroup


class ScoreContext(BaseModel):
    rubric: Rubric
    hypothesis: str
    notes: Optional[str] = None


class Scorer(ABC):
    """The fitness function. It runs only inside ``commit_attempt``, against the
    rubric the turn matched, and is the one judgement the agent cannot argue
    with."""

    source_name: ClassVar[str]
    enabled: bool

    @abstractmethod
    async def score(self, artifact: str, context: ScoreContext) -> Score:
        raise NotImplementedError


class ScorerGroup(DependencyGroup[Scorer]):
    __collects__ = Scorer
