from typing import Dict, List, Literal, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.notes.notes_service import NotesService
from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillReadError,
    SkillResourceNotFoundError,
    SkillsSource,
    SkillWriteError,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

SkillNoteScope = Literal["own", "shared_with_me"]


class SkillNoteSkillsSourceConfiguration(BaseModel):
    enabled: bool = False
    include_scopes: List[SkillNoteScope] = ["own"]
    allowed_skill_note_ids: Optional[List[str]] = None


class SkillNoteSkillsSource(SkillsSource):
    source_name = "skill_notes"

    def __init__(
        self,
        notes_service: NotesService,
        enabled: bool,
        include_scopes: List[SkillNoteScope] = ["own"],
        allowed_skill_note_ids: Optional[List[str]] = None,
    ):
        self.enabled = enabled
        self.__notes_service = notes_service
        self.__include_scopes = include_scopes
        self.__allowed_skill_note_ids = (
            set(allowed_skill_note_ids) if allowed_skill_note_ids else None
        )
        self.__skills: Dict[str, SkillProperties] = {}
        self.__skill_contents: Dict[str, str] = {}
        self.__discovered = False

    async def __ensure_discovered(self) -> None:
        if self.__discovered:
            return
        self.__discovered = True

        all_notes: List = []

        if "own" in self.__include_scopes:
            own_notes = await self.__notes_service.retrieve_notes(
                note_object_type="skill",
                page=1,
                page_size=100,
                created_by_me=True,
            )
            all_notes.extend(own_notes)

        if "shared_with_me" in self.__include_scopes:
            shared_notes = await self.__notes_service.retrieve_notes(
                note_object_type="skill",
                page=1,
                page_size=100,
                created_by_me=False,
            )
            all_notes.extend(shared_notes)

        self.__ingest_notes(all_notes)

    def __ingest_notes(self, notes: List) -> None:
        for note in notes:
            note_id = note.get("id")
            note_id_str = str(note_id) if note_id else None

            if self.__allowed_skill_note_ids is not None and (
                note_id_str is None or note_id_str not in self.__allowed_skill_note_ids
            ):
                continue

            content = note.get("content", "")
            if not content:
                continue
            try:
                properties = self.parse_skill_properties(content)
            except SkillReadError as e:
                logger.warning(f"Skipping skill note: {e}")
                continue

            if properties.name in self.__skills:
                logger.warning(f"Duplicate skill '{properties.name}', keeping first")
                continue

            note_meta: Dict[str, str] = {}
            note_id = note.get("id")
            if note_id:
                note_meta["note_id"] = str(note_id)
            permission = note.get("permission")
            if permission:
                note_meta["permission"] = str(permission)
            if note_meta:
                properties.metadata = {**(properties.metadata or {}), **note_meta}

            self.__skills[properties.name] = properties
            self.__skill_contents[properties.name] = content
            logger.info(f"Loaded skill from note: {properties.name}")

    async def discover(self) -> Dict[str, SkillProperties]:
        await self.__ensure_discovered()
        return dict(self.__skills)

    async def get_skill_content(self, skill_name: str) -> str:
        await self.__ensure_discovered()

        content = self.__skill_contents.get(skill_name)
        if content is None:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))

        return content

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
        properties = self.parse_skill_properties(content)

        await self.__ensure_discovered()
        if properties.name in self.__skills:
            raise SkillWriteError(
                f"Skill '{properties.name}' already exists in source "
                f"'{self.source_name}'."
            )

        try:
            result = await self.__notes_service.create_note(
                content=content, object_type="skill"
            )
        except Exception as e:
            raise SkillWriteError(
                f"Failed to create skill note for skill '{name}': {e}"
            ) from e

        note_id = result.get("id")
        if note_id:
            existing_metadata = properties.metadata or {}
            properties.metadata = {**existing_metadata, "note_id": str(note_id)}

        self.__skills[properties.name] = properties
        self.__skill_contents[properties.name] = content

        logger.info(f"Created skill note: {properties.name}")
        return properties

    async def update_skill(self, name: str, content: str) -> SkillProperties:
        properties = self.parse_skill_properties(content)

        await self.__ensure_discovered()
        existing = self.__skills.get(name)
        if not existing:
            raise SkillWriteError(
                f"Skill '{name}' not found in source '{self.source_name}'."
            )

        metadata = existing.metadata or {}
        permission = metadata.get("permission", "").upper()
        if permission == "READ":
            raise SkillWriteError(
                f"Skill '{name}' is read-only — you don't have permission "
                f"to edit it."
            )

        note_id = metadata.get("note_id")
        if not note_id:
            raise SkillWriteError(
                f"Skill '{name}' cannot be updated — no writable reference."
            )

        try:
            numeric_note_id = int(note_id)
        except (ValueError, TypeError):
            raise SkillWriteError(f"Skill '{name}' has an invalid note reference.")

        try:
            await self.__notes_service.update_note(
                note_id=numeric_note_id,
                content=content,
                object_type="skill",
            )
        except Exception as e:
            raise SkillWriteError(f"Failed to update skill '{name}': {e}") from e

        properties.metadata = {
            **(properties.metadata or {}),
            "note_id": note_id,
            "permission": permission or "WRITE",
        }
        self.__skills[name] = properties
        self.__skill_contents[name] = content

        logger.info(f"Updated skill note: {properties.name}")
        return properties


class SkillNoteSkillsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        notes_service: NotesService,
        skill_note_skills_source_configuration: SkillNoteSkillsSourceConfiguration = (
            SkillNoteSkillsSourceConfiguration()
        ),
    ) -> SkillNoteSkillsSource:
        config = skill_note_skills_source_configuration
        return SkillNoteSkillsSource(
            notes_service=notes_service,
            enabled=config.enabled,
            include_scopes=config.include_scopes,
            allowed_skill_note_ids=config.allowed_skill_note_ids,
        )
