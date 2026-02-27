from typing import Dict, List

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.notes.notes_service import NotesService
from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillReadError,
    SkillResourceNotFoundError,
    SkillsSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class TagNoteSkillsSourceConfig(BaseModel):
    """Configuration for tag-note-backed skills."""

    tag_ids: List[str] = Field(
        default_factory=list,
        description="Tag IDs to load skills from.",
    )


class TagNoteSkillsSource(SkillsSource):
    """Skills source backed by tag notes from the saved-results service.

    Each tag note is expected to contain text in the standard SKILL.md
    format (YAML frontmatter + markdown body).
    """

    source_name = "tag_notes"

    def __init__(self, tag_ids: List[str], notes_service: NotesService):
        self.__tag_ids = tag_ids
        self.__notes_service = notes_service
        self.__skills: Dict[str, SkillProperties] = {}
        self.__skill_contents: Dict[str, str] = {}
        self.__discovered = False

    async def __ensure_discovered(self) -> None:
        if self.__discovered:
            return
        self.__discovered = True

        for tag_id in self.__tag_ids:
            notes = await self.__notes_service.retrieve_tag_notes(
                tag_type="own",
                tag_id=int(tag_id),
                page=1,
                page_size=100,
            )
            for note in notes:
                content = note.get("content", "")
                if not content:
                    continue
                try:
                    properties = self.parse_skill_properties(content)
                except SkillReadError as e:
                    logger.warning(f"Skipping tag note in tag {tag_id}: {e}")
                    continue

                self.__skills[properties.name] = properties
                self.__skill_contents[properties.name] = content
                logger.info(f"Loaded skill from tag note: {properties.name}")

    async def discover(self) -> Dict[str, SkillProperties]:
        await self.__ensure_discovered()
        return dict(self.__skills)

    async def get_skill_body(self, skill_name: str) -> str:
        await self.__ensure_discovered()

        content = self.__skill_contents.get(skill_name)
        if content is None:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))

        return self.parse_skill_body(content)

    async def get_resources(self, skill_name: str) -> List[str]:
        await self.__ensure_discovered()
        if skill_name not in self.__skills:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))
        return []

    async def read_resource(self, skill_name: str, path: str) -> str:
        await self.__ensure_discovered()
        if skill_name not in self.__skills:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))
        raise SkillResourceNotFoundError(skill_name, path, [])


class TagNoteSkillsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notes_service: NotesService,
        tag_note_skills_source_config: TagNoteSkillsSourceConfig = (
            TagNoteSkillsSourceConfig()
        ),
    ) -> TagNoteSkillsSource:
        return TagNoteSkillsSource(
            tag_ids=tag_note_skills_source_config.tag_ids,
            notes_service=notes_service,
        )
