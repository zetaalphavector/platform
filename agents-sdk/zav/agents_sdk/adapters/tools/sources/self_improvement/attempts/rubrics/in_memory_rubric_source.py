from typing import List

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.rubric import Rubric, RubricSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class InMemoryRubricSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Rubrics given inline.")
    rubrics: List[Rubric] = Field(default_factory=list)


class InMemoryRubricSource(RubricSource):
    """Rubrics passed in configuration; a harness that controls the turn hands
    the row's gold answer in through handler params this way."""

    source_name = "in_memory"

    def __init__(self, enabled: bool, rubrics: List[Rubric]) -> None:
        self.enabled = enabled
        self.__rubrics = rubrics

    async def list_rubrics(self) -> List[Rubric]:
        return list(self.__rubrics)


class InMemoryRubricSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        in_memory_rubric_source_configuration: InMemoryRubricSourceConfiguration = (
            InMemoryRubricSourceConfiguration()
        ),
    ) -> InMemoryRubricSource:
        return InMemoryRubricSource(
            enabled=in_memory_rubric_source_configuration.enabled,
            rubrics=in_memory_rubric_source_configuration.rubrics,
        )
