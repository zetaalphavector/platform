from pathlib import Path
from typing import Dict, List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillReadError,
    SkillResourceNotFoundError,
    SkillsSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class DiskSkillsSourceConfig(BaseModel):
    skills_directories: List[str] = Field(
        default_factory=list,
        description="List of directories to scan for skills.",
    )


class DiskSkillsSource(SkillsSource):
    """Default implementation: discovers skills from filesystem directories."""

    source_name = "disk"

    def __init__(self, directories: List[str]):
        self.__directories = directories
        self.__skills: Dict[str, SkillProperties] = {}
        self.__skill_paths: Dict[str, Path] = {}
        self.__skill_bodies: Dict[str, str] = {}
        self.__skill_resources: Dict[str, List[str]] = {}
        self.__discovered = False

    async def __do_discover(self) -> None:
        if self.__discovered:
            return
        self.__discovered = True
        for directory in self.__directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning(f"Skills directory not found: {directory}")
                continue

            for skill_path in dir_path.iterdir():
                if skill_path.is_dir():
                    skill_md = self.__find_skill_md(skill_path)
                    if skill_md:
                        self.__load_skill(skill_path, skill_md)

    def __find_skill_md(self, directory: Path) -> Optional[Path]:
        for path in directory.iterdir():
            if path.is_file() and path.name.lower() == "skill.md":
                return path
        return None

    def __load_skill(self, skill_dir: Path, skill_md: Path) -> None:
        try:
            content = skill_md.read_text()
        except Exception as e:
            logger.warning(f"Failed to read skill file {skill_md}: {e}")
            return

        try:
            properties = self.parse_skill_properties(content)
            body = self.parse_skill_body(content)
        except SkillReadError as e:
            logger.warning(f"Skipping {skill_md}: {e}")
            return

        properties.location = str(skill_md)
        self.__skills[properties.name] = properties
        self.__skill_bodies[properties.name] = body
        self.__skill_paths[properties.name] = skill_dir
        self.__index_resources(properties.name, skill_dir)
        logger.info(f"Loaded skill: {properties.name}")

    def __index_resources(self, skill_name: str, skill_dir: Path) -> None:
        resources: List[str] = []

        for file_path in skill_dir.rglob("*"):
            if file_path.is_file() and file_path.name.lower() != "skill.md":
                relative_path = file_path.relative_to(skill_dir)
                resources.append(str(relative_path))

        self.__skill_resources[skill_name] = sorted(resources)
        if resources:
            logger.info(f"Indexed {len(resources)} resources for skill '{skill_name}'")

    async def discover(self) -> Dict[str, SkillProperties]:
        await self.__do_discover()
        return dict(self.__skills)

    async def get_skill_body(self, skill_name: str) -> str:
        await self.__do_discover()

        if skill_name in self.__skill_bodies:
            return self.__skill_bodies[skill_name]

        if skill_name not in self.__skill_paths:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))

        skill_dir = self.__skill_paths[skill_name]
        skill_md = self.__find_skill_md(skill_dir)
        if not skill_md:
            raise SkillReadError(f"Skill file not found: {skill_name}")

        try:
            content = skill_md.read_text()
        except Exception as e:
            raise SkillReadError(f"Error reading skill file: {e}") from e

        body = self.parse_skill_body(content)
        self.__skill_bodies[skill_name] = body
        return body

    async def get_resources(self, skill_name: str) -> List[str]:
        await self.__do_discover()
        if skill_name not in self.__skills:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))
        return self.__skill_resources.get(skill_name, [])

    async def read_resource(self, skill_name: str, path: str) -> str:
        await self.__do_discover()

        if skill_name not in self.__skill_paths:
            raise SkillNotFoundError(skill_name, list(self.__skills.keys()))

        allowed_resources = self.__skill_resources.get(skill_name, [])
        if path not in allowed_resources:
            raise SkillResourceNotFoundError(skill_name, path, allowed_resources)

        skill_dir = self.__skill_paths[skill_name]
        resource_file = skill_dir / path

        try:
            content = resource_file.read_text()
        except Exception as e:
            raise SkillReadError(f"Error reading resource: {e}") from e

        return content


class DiskSkillsSourceFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        disk_skills_source_config: DiskSkillsSourceConfig = DiskSkillsSourceConfig(),
    ) -> DiskSkillsSource:
        return DiskSkillsSource(
            directories=disk_skills_source_config.skills_directories
        )
