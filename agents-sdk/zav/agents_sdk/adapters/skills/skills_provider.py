import html
from typing import Any, Dict, List, Literal, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillReadError,
    SkillResourceNotFoundError,
    SkillsSource,
    SkillsSourceGroup,
    SkillWriteError,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, ToolStreamingConfig


class SkillsProviderConfiguration(BaseModel):
    """Configuration for skills discovery and presentation."""

    enabled: bool = Field(True, description="Enable the skills system.")
    prompt_format: Literal["xml", "text"] = Field(
        "text", description="Format for skill catalog in system prompt."
    )
    include_location_in_prompt: bool = Field(
        False, description="Include file paths in prompt (for filesystem-based agents)."
    )
    injection_mode: Literal["prompt", "tools", "both"] = Field(
        "both",
        description=(
            "Where to inject skills: 'prompt' (system prompt only), "
            "'tools' (tool definitions only), or 'both'."
        ),
    )
    include_sources: Optional[List[str]] = Field(
        None, description="Allowlist of skill source names to include."
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of skill source names to exclude."
    )
    writable_source: Optional[str] = Field(
        None,
        description=(
            "source_name of the source to write new skills to. "
            "When None, the first registered source is used."
        ),
    )


class SkillsProvider:
    """
    Progressive disclosure of agent skills following the Agent Skills protocol.

    Delegates discovery to one or more SkillsSource instances and handles tool
    generation and prompt formatting.
    """

    def __init__(
        self,
        sources: List[SkillsSource],
        enabled: bool,
        prompt_format: Literal["xml", "text"],
        include_location_in_prompt: bool,
        injection_mode: Literal["prompt", "tools", "both"],
        include_sources: Optional[Set[str]] = None,
        exclude_sources: Optional[Set[str]] = None,
        writable_source: Optional[str] = None,
    ):
        self.__sources = sources
        self.__enabled = enabled
        self.__prompt_format = prompt_format
        self.__include_location = include_location_in_prompt
        self.__injection_mode = injection_mode
        self.__include_sources = include_sources
        self.__exclude_sources = exclude_sources
        self.__skills: Optional[Dict[str, SkillProperties]] = None
        self.__skill_sources: Dict[str, SkillsSource] = {}
        self.__writable_source = self.__resolve_writable(sources, writable_source)
        self.__cached_active: Optional[List[SkillsSource]] = None
        self.__created_skills: List[SkillProperties] = []
        self.__edited_skills: List[SkillProperties] = []

    def __active_sources(self) -> List[SkillsSource]:
        if self.__cached_active is not None:
            return self.__cached_active
        self.__cached_active = [
            source
            for source in self.__sources
            if is_source_active(
                source.source_name,
                source.enabled,
                self.__include_sources,
                self.__exclude_sources,
            )
        ]
        return self.__cached_active

    def describe_loaded(self) -> Dict[str, Any]:
        return {
            "enabled": self.__enabled,
            "sources": [s.source_name for s in self.__active_sources()],
        }

    async def __ensure_discovered(self) -> Dict[str, SkillProperties]:
        if self.__skills is not None:
            return self.__skills
        self.__skills = {}
        for source in self.__active_sources():
            try:
                skills = await source.discover()
            except Exception as e:
                logger.error(
                    f"Failed to discover skills from '{source.source_name}': {e}"
                )
                continue
            for name, props in skills.items():
                if name in self.__skills:
                    logger.warning(
                        f"Skill '{name}' already registered"
                        f" by '{self.__skill_sources[name].source_name}',"
                        f" ignoring duplicate from"
                        f" '{source.source_name}'"
                    )
                    continue
                self.__skills[name] = props
                self.__skill_sources[name] = source
        return self.__skills

    async def to_prompt(self, format: Optional[Literal["xml", "text"]] = None) -> str:
        """Generate skill catalog for system prompt injection."""
        if not self.__enabled:
            return ""
        if self.__injection_mode not in ("prompt", "both"):
            return ""

        skills = await self.__ensure_discovered()
        if not skills:
            return ""
        fmt = format or self.__prompt_format

        if fmt == "xml":
            return self.__to_xml_prompt(skills)
        else:
            return self.__to_text_prompt(skills)

    def __to_xml_prompt(self, skills: Dict[str, SkillProperties]) -> str:
        lines = ["<available_skills>"]

        for skill in skills.values():
            lines.append("<skill>")
            lines.append("<name>")
            lines.append(html.escape(skill.name))
            lines.append("</name>")
            lines.append("<description>")
            lines.append(html.escape(skill.description))
            lines.append("</description>")

            if self.__include_location and skill.location:
                lines.append("<location>")
                lines.append(skill.location)
                lines.append("</location>")

            lines.append("</skill>")

        lines.append("</available_skills>")
        return "\n".join(lines)

    def __to_text_prompt(self, skills: Dict[str, SkillProperties]) -> str:
        lines = ["Available Skills:"]
        lines.append("")

        for skill in skills.values():
            lines.append(f"- **{skill.name}**: {skill.description}")

        lines.append("")
        lines.append("To use a skill, call the `use_skill` tool with the skill name.")

        return "\n".join(lines)

    async def get_tools(self) -> List[Tool]:
        """Return all skill-related tools for registration with ToolsRegistry."""
        if not self.__enabled:
            return []
        if self.__injection_mode not in ("tools", "both"):
            return []

        skills = await self.__ensure_discovered()

        tools = []
        skill_tool = self.__get_skill_tool(skills)
        if skill_tool:
            tools.append(skill_tool)
        resource_tool = await self.__get_resource_tool(skills)
        if resource_tool:
            tools.append(resource_tool)
        content_tool = self.__get_skill_content_tool(skills)
        if content_tool:
            tools.append(content_tool)
        create_tool = self.__get_create_skill_tool()
        if create_tool:
            tools.append(create_tool)
        edit_tool = self.__get_edit_skill_content_tool(skills)
        if edit_tool:
            tools.append(edit_tool)
        return tools

    def __get_skill_tool(self, skills: Dict[str, SkillProperties]) -> Optional[Tool]:
        if not skills:
            return None

        desc_lines = [
            "Activate a skill to receive detailed instructions for a specialized task.",
            "",
            "Available skills:",
        ]
        for skill in skills.values():
            desc_lines.append(f"- {skill.name}: {skill.description}")

        return Tool(
            name="use_skill",
            description="\n".join(desc_lines),
            executable=self.__invoke_skill,
            parameters_spec={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to activate",
                        "enum": list(skills.keys()),
                    }
                },
                "required": ["skill_name"],
            },
            streaming_config=ToolStreamingConfig(
                running_text="Reading skill {{ skill_name }}...",
                completed_text="Read skill {{ skill_name }}",
            ),
        )

    async def __invoke_skill(self, skill_name: str) -> str:
        try:
            skills = await self.__ensure_discovered()
            skill = skills.get(skill_name)
            if not skill:
                raise SkillNotFoundError(skill_name, list(skills.keys()))

            source = self.__skill_sources[skill_name]
            body = await source.get_skill_body(skill_name)
            resources = await source.get_resources(skill_name)

            result = f"# Skill: {skill.name}\n\n"
            result += f"**Description**: {skill.description}\n\n"

            if skill.allowed_tools:
                result += f"**Allowed Tools**: {skill.allowed_tools}\n\n"

            if resources:
                result += f"**Available Resources**: {', '.join(resources)}\n\n"
                result += (
                    "Use `read_skill_resource` tool to load any of these files.\n\n"
                )

            result += "## Instructions\n\n"
            result += body

            return result
        except (SkillNotFoundError, SkillReadError) as e:
            return f"Error: {e}"

    async def __get_resource_tool(
        self, skills: Dict[str, SkillProperties]
    ) -> Optional[Tool]:
        all_resources: Dict[str, List[str]] = {}
        for skill_name in skills:
            source = self.__skill_sources[skill_name]
            resources = await source.get_resources(skill_name)
            if resources:
                all_resources[skill_name] = resources

        if not all_resources:
            return None

        desc_lines = [
            "Read a resource file from a skill's directory.",
            "",
            "Available resources (format: skill_name:path):",
        ]
        for skill_name, resources in all_resources.items():
            desc_lines.append(f"  {skill_name}:")
            for resource in resources:
                desc_lines.append(f"    - {resource}")

        return Tool(
            name="read_skill_resource",
            description="\n".join(desc_lines),
            executable=self.__read_resource,
            parameters_spec={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill",
                        "enum": list(all_resources.keys()),
                    },
                    "resource_path": {
                        "type": "string",
                        "description": "Path to the resource file within the skill",
                    },
                },
                "required": ["skill_name", "resource_path"],
            },
        )

    async def __read_resource(self, skill_name: str, resource_path: str) -> str:
        try:
            source = self.__skill_sources.get(skill_name)
            if source is None:
                skills = await self.__ensure_discovered()
                raise SkillNotFoundError(skill_name, list(skills.keys()))
            content = await source.read_resource(skill_name, resource_path)
            return f"# Resource: {skill_name}/{resource_path}\n\n{content}"
        except (SkillNotFoundError, SkillResourceNotFoundError, SkillReadError) as e:
            return f"Error: {e}"

    @staticmethod
    def __resolve_writable(
        sources: List[SkillsSource],
        writable_source: Optional[str],
    ) -> Optional[SkillsSource]:
        if not sources:
            return None
        if writable_source:
            for source in sources:
                if source.source_name == writable_source:
                    return source
            return None
        return sources[0]

    def __get_skill_content_tool(
        self, skills: Dict[str, SkillProperties]
    ) -> Optional[Tool]:
        if not skills:
            return None

        return Tool(
            name="get_skill_content",
            description=(
                "Read the full raw content of a skill (YAML frontmatter + "
                "markdown body). Use this before calling edit_skill_content "
                "so you have the exact current text to base your edits on."
            ),
            executable=self.__get_skill_content,
            parameters_spec={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to read.",
                    },
                },
                "required": ["name"],
            },
            streaming_config=ToolStreamingConfig(
                running_text="Reading skill content {{ name }}...",
                completed_text="Read skill content {{ name }}",
            ),
        )

    async def __get_skill_content(self, name: str) -> str:
        source = self.__skill_sources.get(name)
        if not source:
            skills = await self.__ensure_discovered()
            return (
                f"Error: skill '{name}' not found. Available: "
                f"{', '.join(skills.keys())}"
            )

        try:
            return await source.get_skill_content(name)
        except (SkillReadError, SkillNotFoundError) as e:
            return f"Error: {e}"

    def __get_create_skill_tool(self) -> Optional[Tool]:
        if not self.__writable_source:
            return None
        if self.__writable_source not in self.__active_sources():
            return None

        return Tool(
            name="create_skill",
            description=(
                "Create a new reusable skill. The content must be a valid "
                "SKILL.md file with YAML frontmatter (name, description) "
                "followed by a markdown instruction body."
            ),
            executable=self.__create_skill,
            parameters_spec={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Skill identifier (lowercase, hyphen-separated). "
                            "Must match the 'name' field in frontmatter."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Full SKILL.md content: YAML frontmatter "
                            "(--- delimited) with at least 'name' and "
                            "'description', followed by markdown body."
                        ),
                    },
                },
                "required": ["name", "content"],
            },
            streaming_config=ToolStreamingConfig(
                running_text="Creating skill {{ name }}...",
                completed_text="Created skill {{ name }}",
                response_transform=lambda r: (
                    {**r, "metadata": self.__created_skills[-1].metadata}
                    if r and self.__created_skills
                    else r
                ),
            ),
        )

    async def __create_skill(self, name: str, content: str) -> Dict:
        try:
            properties = SkillsSource.parse_skill_properties(content)
            if properties.name != name:
                return {
                    "error": (
                        f"Skill name mismatch — parameter 'name' is "
                        f"'{name}' but frontmatter 'name' is '{properties.name}'. "
                        f"They must match."
                    )
                }
        except SkillReadError as e:
            return {"error": f"Invalid SKILL.md content — {e}"}

        source = self.__writable_source
        if not source:
            return {"error": "No writable skill source configured."}

        try:
            saved = await source.create_skill(name, content)
        except SkillWriteError as e:
            return {"error": str(e)}

        self.__skills = None
        self.__skill_sources.clear()
        self.__created_skills.append(saved)

        return {
            "info": (
                f"Skill '{saved.name}' created successfully "
                f"in source '{source.source_name}'.\n\n"
                f"Description: {saved.description}"
            ),
        }

    def __get_edit_skill_content_tool(
        self, skills: Dict[str, SkillProperties]
    ) -> Optional[Tool]:
        if not self.__writable_source:
            return None
        if self.__writable_source not in self.__active_sources():
            return None
        if not skills:
            return None

        return Tool(
            name="edit_skill_content",
            description=(
                "Edit an existing skill using find-and-replace. You MUST "
                "first read the skill's full content using "
                "get_skill_content before calling this. "
                "Provide a list of edits, each with the exact text to "
                "find (old_str) and its replacement (new_str). Each "
                "old_str must match exactly once; if it appears multiple "
                "times, include more surrounding context. Never call "
                "this tool without reading the skill content first."
            ),
            executable=self.__edit_skill_content,
            parameters_spec={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": ("Name of the existing skill to update."),
                    },
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_str": {
                                    "type": "string",
                                    "description": (
                                        "Exact text to find in the skill "
                                        "content. Must match exactly once."
                                    ),
                                },
                                "new_str": {
                                    "type": "string",
                                    "description": ("Text to replace old_str with."),
                                },
                            },
                            "required": ["old_str", "new_str"],
                        },
                        "description": (
                            "List of find-and-replace edits to apply " "sequentially."
                        ),
                    },
                },
                "required": ["name", "edits"],
            },
            streaming_config=ToolStreamingConfig(
                running_text="Updating skill {{ name }}...",
                completed_text="Updated skill {{ name }}",
                response_transform=lambda r: (
                    {**r, "metadata": self.__edited_skills[-1].metadata}
                    if r and self.__edited_skills
                    else r
                ),
            ),
        )

    async def __edit_skill_content(
        self, name: str, edits: List[Dict[str, str]]
    ) -> Dict:
        source = self.__skill_sources.get(name)
        if not source:
            return {"error": f"Skill '{name}' not found."}

        try:
            content = await source.get_skill_content(name)
        except (SkillReadError, SkillNotFoundError) as e:
            return {"error": str(e)}

        if not edits:
            return {"error": "No edits provided."}

        for i, edit in enumerate(edits):
            old_str = edit.get("old_str", "")
            new_str = edit.get("new_str", "")

            if not old_str:
                return {
                    "error": (
                        f"Edit {i + 1}/{len(edits)} failed: "
                        f"old_str must not be empty. "
                        f"No edits were applied."
                    )
                }

            count = content.count(old_str)
            preview = old_str[:80] + "…" if len(old_str) > 80 else old_str

            if count == 0:
                return {
                    "error": (
                        f"Edit {i + 1}/{len(edits)} failed for "
                        f"old_str='{preview}': "
                        f"this text was not found in the skill content. "
                        f"Make sure you've read the skill first and the "
                        f"text matches exactly. "
                        f"No edits were applied."
                    )
                }

            if count > 1:
                return {
                    "error": (
                        f"Edit {i + 1}/{len(edits)} failed for "
                        f"old_str='{preview}': "
                        f"this text appears {count} times in the skill "
                        f"— include more surrounding context to make "
                        f"the match unique. "
                        f"No edits were applied."
                    )
                }

            content = content.replace(old_str, new_str, 1)

        try:
            properties = SkillsSource.parse_skill_properties(content)
            if properties.name != name:
                return {
                    "error": (
                        f"After edits, skill name changed from "
                        f"'{name}' to '{properties.name}'. "
                        f"The skill name must remain the same."
                    )
                }
        except SkillReadError as e:
            return {"error": (f"After applying edits, skill content is invalid: {e}")}

        try:
            saved = await source.update_skill(name, content)
        except SkillWriteError as e:
            return {"error": str(e)}

        self.__skills = None
        self.__skill_sources.clear()
        self.__edited_skills.append(saved)

        return {
            "info": (
                f"Skill '{saved.name}' updated successfully "
                f"in source '{source.source_name}'."
            ),
        }


class SkillsProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        skills_source_group: SkillsSourceGroup = SkillsSourceGroup(items=[]),
        skills_provider_configuration: SkillsProviderConfiguration = (
            SkillsProviderConfiguration()
        ),
    ) -> SkillsProvider:
        return SkillsProvider(
            sources=skills_source_group.items,
            enabled=skills_provider_configuration.enabled,
            prompt_format=skills_provider_configuration.prompt_format,
            include_location_in_prompt=(
                skills_provider_configuration.include_location_in_prompt
            ),
            injection_mode=skills_provider_configuration.injection_mode,
            include_sources=(
                set(skills_provider_configuration.include_sources)
                if skills_provider_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(skills_provider_configuration.exclude_sources)
                if skills_provider_configuration.exclude_sources
                else None
            ),
            writable_source=skills_provider_configuration.writable_source,
        )
