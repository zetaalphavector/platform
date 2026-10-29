import datetime
import uuid
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Literal, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.rubric import Rubric
from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyFactory,
    DependencyGroup,
)

Verdict = Literal["accepted", "rejected"]
Transcript = List[Dict[str, Any]]


def _new_attempt_id() -> str:
    return uuid.uuid4().hex[:8]


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Score(BaseModel):
    scorer: str = Field(..., description="source_name of the scorer that produced it.")
    value: float
    max_value: float = 1.0
    passed: bool
    details: Optional[str] = Field(
        None,
        description=(
            "Shown to the agent with the verdict; must not describe the gold "
            "(spread, threshold, what the parser read)."
        ),
    )
    reasons: Optional[str] = Field(
        None,
        description=(
            "The grader's reasoning. Stored with the attempt for people reading the "
            "lineage; never returned to the agent, since it paraphrases the reference."
        ),
    )

    def render(self) -> str:
        return f"{self.value:g}/{self.max_value:g}"


class Attempt(BaseModel):
    id: str = Field(default_factory=_new_attempt_id)
    parent_id: Optional[str] = None
    created_at: str = Field(default_factory=_now)
    hypothesis: str
    artifact: str
    notes: Optional[str] = None
    score: Optional[Score] = None
    verdict: Verdict
    has_transcript: bool = False


class AttemptStore(ABC):
    """Durable lineages of scored attempts, one per task name."""

    source_name: ClassVar[str]
    enabled: bool

    @abstractmethod
    async def list_attempts(self, task: str) -> List[Attempt]:
        raise NotImplementedError

    @abstractmethod
    async def get(self, task: str, attempt_id: str) -> Optional[Attempt]:
        raise NotImplementedError

    @abstractmethod
    async def add(self, task: str, attempt: Attempt) -> Attempt:
        raise NotImplementedError

    @abstractmethod
    async def save_transcript(
        self, task: str, attempt_id: str, transcript: Transcript
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def load_transcript(self, task: str, attempt_id: str) -> Optional[Transcript]:
        raise NotImplementedError


class AttemptStoreGroup(DependencyGroup[AttemptStore]):
    __collects__ = AttemptStore


def best_accepted(attempts: List[Attempt]) -> Optional[Attempt]:
    accepted = [a for a in attempts if a.verdict == "accepted" and a.score is not None]
    if not accepted:
        return None
    return max(accepted, key=lambda a: (a.score.value, a.created_at))  # type: ignore


class AttemptLineage:
    """The scored attempts of the task the current turn is about, shared across
    the agent build the way ``CitationStore`` is: the attempts provider selects
    the task when the turn's question matches a rubric and commits into it; the
    plateau and transcript-recorder interjection sources read it. Backed by the
    one enabled ``AttemptStore``; without a store, or before a task is selected,
    it is inert."""

    def __init__(self, store: Optional[AttemptStore]) -> None:
        self.__store = store
        self.__rubric: Optional[Rubric] = None

    @property
    def enabled(self) -> bool:
        return self.__store is not None

    @property
    def store_name(self) -> Optional[str]:
        return self.__store.source_name if self.__store else None

    @property
    def rubric(self) -> Optional[Rubric]:
        return self.__rubric

    @property
    def task(self) -> Optional[str]:
        return self.__rubric.name if self.__rubric else None

    @property
    def active(self) -> bool:
        return self.__store is not None and self.__rubric is not None

    def select(self, rubric: Optional[Rubric]) -> None:
        """Called once per turn by the rubric context source, before any provider
        builds its prompt or tools."""
        self.__rubric = rubric

    async def list_attempts(self) -> List[Attempt]:
        if not self.active:
            return []
        return await self.__store.list_attempts(self.task)  # type: ignore

    async def get(self, attempt_id: str) -> Optional[Attempt]:
        if not self.active:
            return None
        return await self.__store.get(self.task, attempt_id)  # type: ignore

    async def add(self, attempt: Attempt) -> Attempt:
        if not self.active:
            raise RuntimeError("No attempt store is enabled or no task is selected.")
        return await self.__store.add(self.task, attempt)  # type: ignore

    async def save_transcript(self, attempt_id: str, transcript: Transcript) -> None:
        if self.__store is not None and self.task is not None:
            await self.__store.save_transcript(self.task, attempt_id, transcript)


class AttemptLineageFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls, attempt_store_group: AttemptStoreGroup = AttemptStoreGroup(items=[])
    ) -> AttemptLineage:
        enabled = [store for store in attempt_store_group.items if store.enabled]
        return AttemptLineage(store=enabled[0] if enabled else None)
