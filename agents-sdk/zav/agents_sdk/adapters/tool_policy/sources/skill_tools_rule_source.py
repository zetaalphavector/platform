from typing import List, Optional, Set

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion
from zav.agents_sdk.adapters.skills.skills_provider import (
    USE_SKILL_TOOL_NAME,
    SkillsProvider,
)
from zav.agents_sdk.adapters.tool_policy.tool_policy_source import (
    ToolPolicySource,
    matches_any,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool


def _read_skills(transcript: List[ChatCompletion]) -> Set[str]:
    skills: Set[str] = set()
    for completion in transcript:
        for request in completion.tool_call_requests or []:
            call = request.function_call_request
            if call.name == USE_SKILL_TOOL_NAME:
                skill_name = (call.params or {}).get("skill_name")
                if isinstance(skill_name, str):
                    skills.add(skill_name)
    return skills


class SkillToolsRuleSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Hide a tool named in a skill's `allowed-tools` until the skill is read."
        ),
    )


class SkillToolsRuleSource(ToolPolicySource):
    """Progressive disclosure: a tool named in a skill's `allowed-tools` stays
    hidden until the conversation has read that skill. Decided from the
    transcript, so a resumed or replayed turn decides the same."""

    source_name = "skill_tools"
    source_description = "Tools a skill names stay hidden until the skill is read."

    def __init__(self, skills_provider: SkillsProvider, enabled: bool):
        self.enabled = enabled
        self.__skills_provider = skills_provider

    async def apply(
        self, tools: List[Tool], transcript: Optional[List[ChatCompletion]]
    ) -> List[Tool]:
        if transcript is None:
            return tools
        skill_tools = await self.__skills_provider.allowed_tools_by_skill()
        if not skill_tools:
            return tools
        read = _read_skills(transcript)
        gated = [pattern for patterns in skill_tools.values() for pattern in patterns]
        unlocked = [
            pattern
            for skill, patterns in skill_tools.items()
            if skill in read
            for pattern in patterns
        ]
        return [
            tool
            for tool in tools
            if not matches_any(tool.name, gated) or matches_any(tool.name, unlocked)
        ]


class SkillToolsRuleSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        skills_provider: SkillsProvider,
        skill_tools_rule_source_configuration: SkillToolsRuleSourceConfiguration = (
            SkillToolsRuleSourceConfiguration()
        ),
    ) -> SkillToolsRuleSource:
        return SkillToolsRuleSource(
            skills_provider, enabled=skill_tools_rule_source_configuration.enabled
        )
