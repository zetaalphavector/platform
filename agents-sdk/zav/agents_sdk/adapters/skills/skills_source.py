import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field


class SkillNotFoundError(Exception):
    """Raised when a requested skill does not exist."""

    def __init__(self, skill_name: str, available: List[str]):
        self.skill_name = skill_name
        self.available = available
        super().__init__(
            f"Unknown skill '{skill_name}'. Available: {', '.join(available)}"
        )


class SkillResourceNotFoundError(Exception):
    """Raised when a requested resource does not exist within a skill."""

    def __init__(self, skill_name: str, resource_path: str, available: List[str]):
        self.skill_name = skill_name
        self.resource_path = resource_path
        self.available = available
        msg = f"Resource '{resource_path}' not found in skill '{skill_name}'."
        if available:
            msg += f" Available: {', '.join(available)}"
        super().__init__(msg)


class SkillReadError(Exception):
    """Raised when reading a skill or resource fails."""

    def __init__(self, message: str):
        super().__init__(message)


class SkillProperties(BaseModel):
    """Metadata extracted from SKILL.md frontmatter."""

    name: str = Field(..., description="Unique skill identifier.")
    description: str = Field(..., description="What the skill does and when to use it.")
    license: Optional[str] = Field(
        None, description="License name or reference to bundled license file."
    )
    compatibility: Optional[str] = Field(
        None, description="Environment requirements (product, packages, network, etc.)."
    )
    allowed_tools: Optional[str] = Field(
        None, description="Space-delimited list of pre-approved tools."
    )
    metadata: Optional[Dict[str, str]] = Field(
        None, description="Arbitrary key-value mapping for additional metadata."
    )
    location: Optional[str] = Field(
        None, description="Location hint for filesystem-based agents."
    )


class SkillsSource(ABC):
    """Abstract source for skill discovery. Extend this for custom backends."""

    @abstractmethod
    async def discover(self) -> Dict[str, SkillProperties]:
        """Return all available skills with their metadata."""
        raise NotImplementedError

    @abstractmethod
    async def get_skill_body(self, skill_name: str) -> str:
        """Return the full instruction body for a skill."""
        raise NotImplementedError

    @abstractmethod
    async def get_resources(self, skill_name: str) -> List[str]:
        """Return list of available resource paths for a skill."""
        raise NotImplementedError

    @abstractmethod
    async def read_resource(self, skill_name: str, path: str) -> str:
        """Read the content of a skill resource file."""
        raise NotImplementedError


class DiskSkillsSource(SkillsSource):
    """Default implementation: discovers skills from filesystem directories."""

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

        match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
        if not match:
            logger.warning(f"Invalid SKILL.md format (no frontmatter): {skill_md}")
            return

        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML frontmatter in {skill_md}: {e}")
            return

        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not name or not description:
            logger.warning(f"Missing required fields (name, description) in {skill_md}")
            return

        self.__skills[name] = SkillProperties(
            name=name,
            description=description,
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            allowed_tools=frontmatter.get("allowed-tools"),
            metadata=frontmatter.get("metadata"),
            location=str(skill_md),
        )
        self.__skill_paths[name] = skill_dir
        self.__index_resources(name, skill_dir)
        logger.info(f"Loaded skill: {name}")

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

        match = re.match(r"^---\n.*?\n---\n(.*)$", content, re.DOTALL)
        body = match.group(1).strip() if match else content

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
