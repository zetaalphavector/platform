from zav.agents_sdk.adapters.tool_policy.sources.skill_tools_rule_source import (
    SkillToolsRuleSource,
    SkillToolsRuleSourceFactory,
)
from zav.agents_sdk.adapters.tool_policy.tool_policy_provider import (
    ToolPolicyProvider,
    ToolPolicyProviderFactory,
    ToolSelection,
)
from zav.agents_sdk.adapters.tool_policy.tool_policy_source import (
    ToolPolicyConfiguration,
    ToolPolicySource,
    ToolPolicySourceGroup,
    ask_first,
    load_on_demand,
    matches_any,
    waive_asking,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry
from zav.agents_sdk.domain.tools import Confirmation, ToolOptions

AgentDependencyRegistry.register(ToolPolicyProviderFactory)
AgentDependencyRegistry.register(SkillToolsRuleSourceFactory)

__all__ = [
    "Confirmation",
    "SkillToolsRuleSource",
    "SkillToolsRuleSourceFactory",
    "ToolOptions",
    "ToolPolicyConfiguration",
    "ToolPolicyProvider",
    "ToolPolicyProviderFactory",
    "ToolPolicySource",
    "ToolPolicySourceGroup",
    "ToolSelection",
    "ask_first",
    "load_on_demand",
    "matches_any",
    "waive_asking",
]
