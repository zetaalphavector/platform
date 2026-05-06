from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple, cast

import strictyaml
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_dependency import DependencyGroup


def _strip_common_indent(lines: List[str]) -> List[str]:
    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    if not indents:
        return ["" for _ in lines]
    min_indent = min(indents)
    return [line[min_indent:] if len(line) >= min_indent else "" for line in lines]


def _is_blank_or_comment(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _indent_level(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_key_value_line(line: str, *, line_no: int) -> Tuple[str, str]:
    if line.startswith(" "):
        raise SkillReadError(
            f"Invalid frontmatter: unexpected indentation on line {line_no}"
        )

    if ":" not in line:
        raise SkillReadError(
            f"Invalid frontmatter: expected 'key: value' on line {line_no}"
        )

    key, rest = line.split(":", 1)
    key = key.strip()
    if not key:
        raise SkillReadError(f"Invalid frontmatter: empty key on line {line_no}")

    return key, rest.lstrip(" ")


def _collect_block_scalar_lines(
    lines: List[str], *, start_at: int
) -> Tuple[List[str], int]:
    content_lines: List[str] = []
    index = start_at
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            content_lines.append("")
            index += 1
            continue
        if _indent_level(line) == 0:
            break
        content_lines.append(line)
        index += 1
    return content_lines, index


def _collect_indented_lines(
    lines: List[str], *, start_at: int
) -> Tuple[List[str], int]:
    indented: List[str] = []
    index = start_at
    while index < len(lines):
        line = lines[index]
        if _is_blank_or_comment(line):
            index += 1
            continue
        if _indent_level(line) == 0:
            break
        indented.append(line)
        index += 1
    return indented, index


def _parse_indented_mapping(entries: List[str], *, parent_key: str) -> Dict[str, str]:
    nested: Dict[str, str] = {}
    for entry in entries:
        stripped = entry.lstrip(" ")
        if ":" not in stripped:
            raise SkillReadError(
                "Invalid frontmatter: expected 'key: value' inside nested mapping "
                f"for '{parent_key}'"
            )
        nested_key, nested_rest = stripped.split(":", 1)
        nested_key = nested_key.strip()
        if not nested_key:
            raise SkillReadError(
                "Invalid frontmatter: empty key inside nested mapping "
                f"for '{parent_key}'"
            )
        if nested_key in nested:
            raise SkillReadError(
                "Invalid frontmatter: duplicate key "
                f"'{nested_key}' inside nested mapping for '{parent_key}'"
            )
        nested[nested_key] = nested_rest.lstrip(" ")
    return nested


def _parse_relaxed_frontmatter(frontmatter: str) -> Dict[str, Any]:
    """Best-effort parser for SKILL.md frontmatter.

    This intentionally supports only a small, predictable subset:
    - Top-level `key: value` pairs (values treated as raw strings)
    - Nested mappings when a top-level key has an empty value (`key:`)
    - Block scalars using `|` or `>` (top-level only)

    The goal is to be robust to common user mistakes like unquoted values
    containing `: ` (e.g. `description: foo: bar`).
    """

    lines = frontmatter.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    result: Dict[str, Any] = {}
    index = 0

    while index < len(lines):
        line = lines[index]
        if _is_blank_or_comment(line):
            index += 1
            continue

        key, value = _parse_key_value_line(line, line_no=index + 1)
        if key in result:
            raise SkillReadError(
                f"Invalid frontmatter: duplicate key '{key}' on line {index + 1}"
            )

        if value in ("|", ">"):
            content_lines, index = _collect_block_scalar_lines(
                lines, start_at=index + 1
            )
            result[key] = "\n".join(_strip_common_indent(content_lines)).rstrip("\n")
            continue

        if (
            value == ""
            and (index + 1) < len(lines)
            and _indent_level(lines[index + 1]) > 0
        ):
            nested_lines, index = _collect_indented_lines(lines, start_at=index + 1)
            result[key] = _parse_indented_mapping(nested_lines, parent_key=key)
            continue

        result[key] = value
        index += 1

    return result


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


class SkillWriteError(Exception):
    """Raised when writing/creating a skill fails."""

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

    source_name: ClassVar[str]
    enabled: bool

    @staticmethod
    def parse_skill_properties(content: str) -> SkillProperties:
        """Extract metadata from SKILL.md-formatted content.

        Parses YAML frontmatter using strictyaml (all values treated as
        strings, avoiding issues with colons or other special characters).

        Returns a ``SkillProperties`` instance with ``location`` set to
        ``None`` — callers should set it to a source-specific value if
        needed.

        Raises:
            SkillReadError: If the content has no valid frontmatter or
                required fields are missing.
        """
        if not content.startswith("---"):
            raise SkillReadError("SKILL.md must start with YAML frontmatter (---)")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise SkillReadError("SKILL.md frontmatter not properly closed with ---")

        frontmatter = parts[1]

        try:
            parsed = strictyaml.load(frontmatter)
            metadata = parsed.data
        except strictyaml.YAMLError as strict_error:
            try:
                metadata = _parse_relaxed_frontmatter(frontmatter)
                logger.warning(
                    "Invalid YAML in SKILL.md frontmatter; "
                    "falling back to relaxed parser. "
                    "Tip: quote values containing ': ' "
                    '(e.g. description: "foo: bar").'
                )
            except SkillReadError as relaxed_error:
                raise SkillReadError(
                    "Invalid YAML in frontmatter. "
                    "Tip: quote values containing ': ' "
                    '(e.g. description: "foo: bar"). '
                    f"StrictYAML error: {strict_error}. "
                    f"Relaxed parse error: {relaxed_error}"
                ) from strict_error

        if not isinstance(metadata, dict):
            raise SkillReadError("SKILL.md frontmatter must be a YAML mapping")

        if "name" not in metadata:
            raise SkillReadError("Missing required field in frontmatter: name")
        if "description" not in metadata:
            raise SkillReadError("Missing required field in frontmatter: description")

        name = metadata["name"]
        description = metadata["description"]

        if not isinstance(name, str) or not name.strip():
            raise SkillReadError("Field 'name' must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise SkillReadError("Field 'description' must be a non-empty string")

        raw_metadata = metadata.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata["metadata"] = {str(k): str(v) for k, v in raw_metadata.items()}

        raw_license = metadata.get("license")
        license_value = (
            raw_license
            if raw_license is None or isinstance(raw_license, str)
            else str(raw_license)
        )

        raw_compatibility = metadata.get("compatibility")
        compatibility_value = (
            raw_compatibility
            if raw_compatibility is None or isinstance(raw_compatibility, str)
            else str(raw_compatibility)
        )

        raw_allowed_tools = metadata.get("allowed-tools")
        allowed_tools_value = (
            raw_allowed_tools
            if raw_allowed_tools is None or isinstance(raw_allowed_tools, str)
            else str(raw_allowed_tools)
        )

        metadata_mapping = (
            cast(Dict[str, str], metadata["metadata"])
            if isinstance(metadata.get("metadata"), dict)
            else None
        )

        return SkillProperties(
            name=name.strip(),
            description=description.strip(),
            license=license_value,
            compatibility=compatibility_value,
            allowed_tools=allowed_tools_value,
            metadata=metadata_mapping,
        )

    @staticmethod
    def parse_skill_body(content: str) -> str:
        """Extract the instruction body from SKILL.md-formatted content.

        Returns the markdown body after the closing ``---`` of the
        frontmatter block.

        Raises:
            SkillReadError: If the content has no valid frontmatter.
        """
        if not content.startswith("---"):
            raise SkillReadError("SKILL.md must start with YAML frontmatter (---)")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise SkillReadError("SKILL.md frontmatter not properly closed with ---")

        return parts[2].strip()

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

    async def get_skill_content(self, skill_name: str) -> str:
        """Return the full raw content (frontmatter + body) for a skill.

        Override in sources that support update. The default raises
        SkillReadError.
        """
        raise SkillReadError(
            f"Source '{self.source_name}' does not support reading raw content."
        )

    async def create_skill(self, name: str, content: str) -> SkillProperties:
        """Persist a new skill.

        Override in writable sources. The default raises SkillWriteError.

        Args:
            name: Skill identifier (becomes directory/note name).
            content: Full SKILL.md content (frontmatter + body).

        Returns:
            The properties parsed from the saved content.
        """
        raise SkillWriteError(
            f"Source '{self.source_name}' is read-only and cannot create skills."
        )

    async def update_skill(self, name: str, content: str) -> SkillProperties:
        """Update an existing skill.

        Override in writable sources. The default raises SkillWriteError.

        Args:
            name: Skill identifier of the existing skill to update.
            content: Full SKILL.md content (frontmatter + body).

        Returns:
            The updated properties parsed from the new content.
        """
        raise SkillWriteError(
            f"Source '{self.source_name}' is read-only and cannot update skills."
        )


class SkillsSourceGroup(DependencyGroup[SkillsSource]):
    __collects__ = SkillsSource
