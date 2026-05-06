from typing import Dict, List

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillResourceNotFoundError,
    SkillsSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class InMemorySkillDefinition(BaseModel):
    name: str = Field(..., description="Unique skill identifier.")
    description: str = Field(..., description="What the skill does and when to use it.")
    body: str = Field(..., description="The instruction body returned by use_skill.")


class InMemorySkillsSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable in-memory skills source.")
    skills: List[InMemorySkillDefinition] = Field(
        default_factory=list,
        description="List of skill definitions to register.",
    )


class InMemorySkillsSource(SkillsSource):
    """A skills source backed by an in-memory list of skill definitions.

    Useful for surfaces that define skills purely through configuration
    (e.g. bot_params or handler_params) without needing files on disk
    or dedicated source classes.
    """

    source_name = "in_memory_skills"

    def __init__(
        self,
        enabled: bool,
        skills: List[InMemorySkillDefinition],
    ):
        self.enabled = enabled
        self.__skills: Dict[str, InMemorySkillDefinition] = {
            skill.name: skill for skill in skills
        }

    async def discover(self) -> Dict[str, SkillProperties]:
        return {
            name: SkillProperties(
                name=skill.name,
                description=skill.description,
            )
            for name, skill in self.__skills.items()
        }

    async def get_skill_body(self, skill_name: str) -> str:
        skill = self.__skills.get(skill_name)
        if skill is None:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))
        return skill.body

    async def get_resources(self, skill_name: str) -> List[str]:
        if skill_name not in self.__skills:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))
        return []

    async def read_resource(self, skill_name: str, path: str) -> str:
        if skill_name not in self.__skills:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))
        raise SkillResourceNotFoundError(skill_name, path, [])


class InMemorySkillsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        in_memory_skills_source_configuration: InMemorySkillsSourceConfiguration = (
            InMemorySkillsSourceConfiguration()
        ),
    ) -> InMemorySkillsSource:
        return InMemorySkillsSource(
            skills=in_memory_skills_source_configuration.skills,
            enabled=in_memory_skills_source_configuration.enabled,
        )
