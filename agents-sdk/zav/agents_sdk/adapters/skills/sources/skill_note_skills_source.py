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
    include_scopes: Optional[List[SkillNoteScope]] = ["own"]
    allowed_skill_note_ids: Optional[List[str]] = None


class SkillSelectionConfiguration(BaseModel):
    """The user's skill picks for one chat, applied on top of the agent's setup.

    Sent as a bot param from the chat's Sources and apps panel. A field left
    unset keeps what the agent's setup says.
    """

    enabled: Optional[bool] = None
    include_shared_skills: Optional[bool] = None
    included_skill_note_ids: Optional[List[str]] = None
    excluded_skill_note_ids: Optional[List[str]] = None


class SkillNoteSkillsSource(SkillsSource):
    source_name = "skill_notes"
    source_description = "Skills users saved as skill notes."

    def __init__(
        self,
        notes_service: NotesService,
        enabled: bool,
        include_scopes: Optional[List[SkillNoteScope]] = ["own"],
        allowed_skill_note_ids: Optional[List[str]] = None,
        excluded_skill_note_ids: Optional[List[str]] = None,
        include_shared_skills: bool = True,
    ):
        self.enabled = enabled
        self.__notes_service = notes_service
        self.__include_scopes = include_scopes or []
        self.__allowed_skill_note_ids = allowed_skill_note_ids or []
        self.__excluded_skill_note_ids = set(excluded_skill_note_ids or [])
        self.__include_shared_skills = include_shared_skills
        self.__skills: Dict[str, SkillProperties] = {}
        self.__skill_contents: Dict[str, str] = {}
        self.__discovered = False

    async def __ensure_discovered(self) -> None:
        if self.__discovered:
            return
        self.__discovered = True

        all_notes: List = []
        own_notes: Optional[List] = None

        if "own" in self.__include_scopes:
            own_notes = await self.__retrieve_own_notes()
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

        loaded_note_ids = {str(n.get("id")) for n in all_notes if n.get("id")}
        allowed_ids = [
            note_id
            for note_id in self.__allowed_skill_note_ids
            if note_id not in loaded_note_ids
            and note_id not in self.__excluded_skill_note_ids
        ]
        if allowed_ids and not self.__include_shared_skills:
            # An agent's picked skills can be shared ones; with shared skills
            # switched off only the user's own picks stay.
            if own_notes is None:
                own_notes = await self.__retrieve_own_notes()
            own_ids = {str(n.get("id")) for n in own_notes if n.get("id")}
            allowed_ids = [note_id for note_id in allowed_ids if note_id in own_ids]

        additional_notes: List = []
        for note_id_str in allowed_ids:
            try:
                note = await self.__notes_service.retrieve_note(
                    note_id=int(note_id_str)
                )
            except (ValueError, TypeError):
                logger.warning(f"Invalid allowed_skill_note_id: {note_id_str}")
                continue
            except Exception as e:
                logger.warning(f"Failed to fetch skill note {note_id_str}: {e}")
                continue
            if note:
                additional_notes.append(note)

        if additional_notes:
            self.__ingest_notes(additional_notes)

    async def __retrieve_own_notes(self) -> List:
        return await self.__notes_service.retrieve_notes(
            note_object_type="skill",
            page=1,
            page_size=100,
            created_by_me=True,
        )

    def __ingest_notes(self, notes: List) -> None:
        for note in notes:
            if str(note.get("id")) in self.__excluded_skill_note_ids:
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
        skill_selection_configuration: SkillSelectionConfiguration = (
            SkillSelectionConfiguration()
        ),
    ) -> SkillNoteSkillsSource:
        config = skill_note_skills_source_configuration
        selection = skill_selection_configuration
        include_scopes = list(config.include_scopes or [])
        if selection.include_shared_skills is False:
            include_scopes = [s for s in include_scopes if s != "shared_with_me"]
        elif (
            selection.include_shared_skills
            # Scopes set means "every skill"; an agent with picked skills has
            # none, and shared skills then only means its picked shared ones.
            and include_scopes
            and "shared_with_me" not in include_scopes
        ):
            include_scopes.append("shared_with_me")
        allowed_skill_note_ids = [
            *(config.allowed_skill_note_ids or []),
            *(selection.included_skill_note_ids or []),
        ]
        return SkillNoteSkillsSource(
            notes_service=notes_service,
            enabled=config.enabled and selection.enabled is not False,
            include_scopes=include_scopes,
            allowed_skill_note_ids=allowed_skill_note_ids or None,
            excluded_skill_note_ids=selection.excluded_skill_note_ids,
            include_shared_skills=selection.include_shared_skills is not False,
        )
