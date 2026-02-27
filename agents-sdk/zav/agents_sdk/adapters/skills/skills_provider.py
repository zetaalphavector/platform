import html
from typing import Dict, List, Literal, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillReadError,
    SkillResourceNotFoundError,
    SkillsSource,
    SkillsSourceGroup,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, ToolStreamingConfig


class SkillsConfiguration(BaseModel):
    """Configuration for skills discovery and presentation."""

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


class SkillsProvider:
    """
    Progressive disclosure of agent skills following the Agent Skills protocol.

    Delegates discovery to one or more SkillsSource instances and handles tool
    generation and prompt formatting.
    """

    def __init__(
        self,
        sources: List[SkillsSource],
        prompt_format: Literal["xml", "text"],
        include_location_in_prompt: bool,
        injection_mode: Literal["prompt", "tools", "both"],
    ):
        self.__sources = sources
        self.__prompt_format = prompt_format
        self.__include_location = include_location_in_prompt
        self.__injection_mode = injection_mode
        self.__skills: Optional[Dict[str, SkillProperties]] = None
        self.__skill_sources: Dict[str, SkillsSource] = {}

    async def __ensure_discovered(self) -> Dict[str, SkillProperties]:
        if self.__skills is not None:
            return self.__skills
        self.__skills = {}
        for source in self.__sources:
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


class SkillsProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        skills_source_group: SkillsSourceGroup = SkillsSourceGroup(items=[]),
        skills_configuration: SkillsConfiguration = SkillsConfiguration(),
    ) -> SkillsProvider:
        return SkillsProvider(
            sources=skills_source_group.items,
            prompt_format=skills_configuration.prompt_format,
            include_location_in_prompt=skills_configuration.include_location_in_prompt,
            injection_mode=skills_configuration.injection_mode,
        )
