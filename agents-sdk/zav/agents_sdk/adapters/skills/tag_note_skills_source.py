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
    SkillWriteError,
)
from zav.agents_sdk.adapters.tags.tags_service import TagsService
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class TagNoteSkillsSourceConfig(BaseModel):
    tag_ids: List[str] = Field(
        default_factory=list,
        description="Tag IDs to load skills from.",
    )
    include_favorites: bool = Field(
        False,
        description="Also load skills from the user's favorites tag.",
    )


class TagNoteSkillsSource(SkillsSource):
    source_name = "tag_notes"

    def __init__(
        self,
        notes_service: NotesService,
        tags_service: TagsService,
        tag_ids: List[str],
        include_favorites: bool,
    ):
        self.__tag_ids = tag_ids
        self.__include_favorites = include_favorites
        self.__notes_service = notes_service
        self.__tags_service = tags_service
        self.__skills: Dict[str, SkillProperties] = {}
        self.__skill_contents: Dict[str, str] = {}
        self.__discovered = False

    async def __ensure_discovered(self) -> None:
        if self.__discovered:
            return
        self.__discovered = True

        for tag_id in self.__tag_ids:
            tag_type = await self.__tags_service.get_tag_type_by_id(int(tag_id))
            if tag_type is None:
                logger.warning(f"Tag {tag_id} not found, skipping")
                continue
            notes = await self.__notes_service.retrieve_tag_notes(
                tag_type=tag_type,
                tag_id=int(tag_id),
                page=1,
                page_size=100,
            )
            self.__ingest_notes(notes, source_label=f"tag {tag_id}")

        if self.__include_favorites:
            notes = await self.__notes_service.retrieve_favorite_tag_notes(
                page=1,
                page_size=100,
            )
            self.__ingest_notes(notes, source_label="favorites")

    def __ingest_notes(self, notes: List, source_label: str) -> None:
        for note in notes:
            content = note.get("content", "")
            if not content:
                continue
            try:
                properties = self.parse_skill_properties(content)
            except SkillReadError as e:
                logger.warning(f"Skipping tag note in {source_label}: {e}")
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

    async def create_skill(self, name: str, content: str) -> SkillProperties:
        if not self.__tag_ids:
            raise SkillWriteError("No tag IDs configured for writing.")

        properties = self.parse_skill_properties(content)

        await self.__ensure_discovered()
        if properties.name in self.__skills:
            raise SkillWriteError(
                f"Skill '{properties.name}' already exists in source "
                f"'{self.source_name}'."
            )

        target_tag_id = self.__tag_ids[0]
        tag_type = await self.__tags_service.get_tag_type_by_id(int(target_tag_id))
        if tag_type is None:
            raise SkillWriteError(f"Tag {target_tag_id} not found.")
        try:
            await self.__notes_service.create_tag_note(
                tag_type=tag_type,
                tag_id=int(target_tag_id),
                content=content,
            )
        except Exception as e:
            raise SkillWriteError(
                f"Failed to create tag note for skill '{name}': {e}"
            ) from e

        self.__skills[properties.name] = properties
        self.__skill_contents[properties.name] = content

        logger.info(f"Created skill: {properties.name} in tag {target_tag_id}")
        return properties


class TagNoteSkillsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notes_service: NotesService,
        tags_service: TagsService,
        tag_note_skills_source_config: TagNoteSkillsSourceConfig = (
            TagNoteSkillsSourceConfig()
        ),
    ) -> TagNoteSkillsSource:
        return TagNoteSkillsSource(
            notes_service=notes_service,
            tags_service=tags_service,
            tag_ids=tag_note_skills_source_config.tag_ids,
            include_favorites=tag_note_skills_source_config.include_favorites,
        )
