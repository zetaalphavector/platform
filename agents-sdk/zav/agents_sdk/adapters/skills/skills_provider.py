import html
import re
from typing import Any, Dict, List, Literal, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import active_sources, pick_writable
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

USE_SKILL_TOOL_NAME = "use_skill"


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
    exclude_skills: Optional[List[str]] = Field(
        None, description="Skill names the agent may not use."
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
        exclude_skills: Optional[Set[str]] = None,
    ):
        self.__enabled = enabled
        self.__exclude_skills = exclude_skills or set()
        self.__prompt_format = prompt_format
        self.__include_location = include_location_in_prompt
        self.__injection_mode = injection_mode
        self.__skills: Optional[Dict[str, SkillProperties]] = None
        # Built once, as reading resources is I/O; cleared with the skills.
        self.__tools: Optional[List[Tool]] = None
        self.__skill_sources: Dict[str, SkillsSource] = {}
        self.__active = active_sources(
            sources, enabled, include_sources, exclude_sources
        )
        self.__writable_source = pick_writable(self.__active, writable_source)
        self.__created_skills: List[SkillProperties] = []
        self.__edited_skills: List[SkillProperties] = []

    @property
    def active_sources(self) -> List[SkillsSource]:
        return self.__active

    @property
    def writable_source(self) -> Optional[SkillsSource]:
        """Where new skills are written, for the agent's `create_skill` tool and
        for anything else in the build that persists skills on its behalf."""
        return self.__writable_source

    def __active_sources(self) -> List[SkillsSource]:
        return self.__active

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
                if name in self.__exclude_skills:
                    continue
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

    async def skills_by_source(self) -> Dict[str, List[str]]:
        by_source: Dict[str, List[str]] = {
            source.source_name: [] for source in self.__active_sources()
        }
        for name in await self.__ensure_discovered():
            by_source[self.__skill_sources[name].source_name].append(name)
        return by_source

    async def allowed_tools_by_skill(self) -> Dict[str, List[str]]:
        """Each skill's `allowed-tools`, split into tool names or patterns."""
        if not self.__enabled:
            return {}
        return {
            name: re.split(r"[\s,]+", skill.allowed_tools.strip())
            for name, skill in (await self.__ensure_discovered()).items()
            if skill.allowed_tools and skill.allowed_tools.strip()
        }

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

        if self.__tools is not None:
            return self.__tools
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
        self.__tools = tools
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
            name=USE_SKILL_TOOL_NAME,
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
        self.__tools = None
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
        self.__tools = None
        self.__skill_sources.clear()
        self.__edited_skills.append(saved)

        return {
            "info": (
                f"Skill '{saved.name}' updated successfully "
                f"in source '{source.source_name}'."
            ),
        }


class SkillsProviderFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls,
        skills_source_group: SkillsSourceGroup = SkillsSourceGroup(items=[]),
        skills_provider_configuration: SkillsProviderConfiguration = (
            SkillsProviderConfiguration()
        ),
    ) -> SkillsProvider:
        configuration = skills_provider_configuration
        return SkillsProvider(
            sources=skills_source_group.items,
            enabled=configuration.enabled,
            prompt_format=configuration.prompt_format,
            include_location_in_prompt=configuration.include_location_in_prompt,
            injection_mode=configuration.injection_mode,
            include_sources=(
                set(configuration.include_sources)
                if configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(configuration.exclude_sources)
                if configuration.exclude_sources
                else None
            ),
            writable_source=configuration.writable_source,
            exclude_skills=set(configuration.exclude_skills or []),
        )
