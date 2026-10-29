from typing import Dict, List, Optional, Tuple

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.attempt_lineage import (
    Attempt,
    AttemptStore,
    Transcript,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class InMemoryAttemptStoreConfiguration(BaseModel):
    enabled: bool = Field(False, description="Keep attempts in process memory only.")


class InMemoryAttemptStore(AttemptStore):
    source_name = "in_memory"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.__attempts: Dict[str, List[Attempt]] = {}
        self.__transcripts: Dict[Tuple[str, str], Transcript] = {}

    async def list_attempts(self, task: str) -> List[Attempt]:
        return [
            a.copy(update={"has_transcript": (task, a.id) in self.__transcripts})
            for a in self.__attempts.get(task, [])
        ]

    async def get(self, task: str, attempt_id: str) -> Optional[Attempt]:
        for attempt in await self.list_attempts(task):
            if attempt.id == attempt_id:
                return attempt
        return None

    async def add(self, task: str, attempt: Attempt) -> Attempt:
        self.__attempts.setdefault(task, []).append(attempt)
        return attempt

    async def save_transcript(
        self, task: str, attempt_id: str, transcript: Transcript
    ) -> None:
        self.__transcripts[(task, attempt_id)] = list(transcript)

    async def load_transcript(self, task: str, attempt_id: str) -> Optional[Transcript]:
        return self.__transcripts.get((task, attempt_id))


class InMemoryAttemptStoreFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls,
        in_memory_attempt_store_configuration: InMemoryAttemptStoreConfiguration = (
            InMemoryAttemptStoreConfiguration()
        ),
    ) -> InMemoryAttemptStore:
        return InMemoryAttemptStore(
            enabled=in_memory_attempt_store_configuration.enabled
        )
