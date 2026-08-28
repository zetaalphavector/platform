from collections import Counter
from typing import Any, Dict, List, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.agent_delegation.delegable_agents_source import (
    DelegableAgent,
    DelegableAgentsSource,
    DelegableAgentsSourceGroup,
)
from zav.agents_sdk.domain.agent_creator import (
    AgentCreator,
    SubAgentToolResult,
    run_sub_agent,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, ToolStreamingConfig

_MINIMAL_DELEGATE_SYSTEM_PROMPT = """\
## Agent delegation

You have access to a `delegate` tool that delegates a task to a specialized agent. \
Each agent has its own tools and capabilities. The target agent runs in a fresh \
context — it cannot see your conversation history — and returns a text result \
as this tool's response. Visible tool outputs created by the delegated agent \
may also appear in this chat.\
"""

_MINIMAL_DELEGATE_TOOL_DESCRIPTION = """\
Delegate a task to a specialized agent. The target agent runs in a fresh context \
with no access to your conversation history. It executes autonomously using its \
own tools and capabilities, returns a text result as this tool's response, and \
may surface tool outputs in this chat.\
"""

_VERBOSE_DELEGATE_SYSTEM_PROMPT = """\
## Agent delegation

You have access to a `delegate` tool that delegates a task to a specialized agent. \
Each agent has its own tools and capabilities. The target agent runs in a fresh \
context — it cannot see your conversation history — and returns a text result \
as this tool's response. Visible tool outputs created by the delegated agent \
may also appear in this chat.

Use `delegate` when a specialized agent is better suited for the task — for example, \
when the user explicitly asks for a specific agent's capabilities, or when a domain \
expert agent would produce a better result than your general-purpose tools.

When delegating:
- Write a SELF-CONTAINED prompt that includes all relevant context.
- Specify the desired output format and length constraints.
- Use the returned text result to answer the user; it may also be visible in the \
delegation tool UI when the chat frontend renders tool details.

Prefer your own tools over delegation when you can handle the task directly.\
"""

_VERBOSE_DELEGATE_TOOL_DESCRIPTION = """\
Delegate a task to a specialized agent. The target agent runs in a fresh context \
with no access to your conversation history. It executes autonomously using its \
own tools and capabilities, returns a text result as this tool's response, and \
may surface visible tool outputs in this chat.

Use this tool when:
- The user's request matches the expertise of a specific agent.
- A specialized agent has tools or knowledge better suited to the task.
- You want to leverage a domain-specific agent for a focused piece of work.

Writing effective prompts:
1. Prompts must be SELF-CONTAINED — include all relevant context the target \
agent needs. It cannot see your conversation.
2. Specify exact output format AND length constraints \
(e.g., "one paragraph, ≤90 words" or "bullet list, max 5 items").
3. Use the returned text result to answer the user; it may also be visible in \
the delegation tool UI when the chat frontend renders tool details.

When NOT to use this tool:
- If you can handle the task yourself with your own tools.
- If the task depends on nuances from your conversation that are hard to \
summarize.\
"""


class AgentDelegationProviderConfiguration(BaseModel):
    """Configuration for the agent delegation provider."""

    enabled: bool = Field(
        False,
        description="Enable agent delegation tool.",
    )
    verbose: bool = Field(
        True,
        description=(
            "Use verbose behavioral guidance in tool description and system prompt."
        ),
    )
    include_in_prompt: bool = Field(
        True,
        description="Also inject behavioral instructions into the system prompt.",
    )
    include_sources: Optional[List[str]] = Field(
        None, description="Allowlist of delegable agent source names to include."
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of delegable agent source names to exclude."
    )
    extra_instructions: Optional[str] = Field(
        None,
        description="Additional instructions appended to the behavioral guidance.",
    )
    tool_description: Optional[str] = Field(
        None,
        description=(
            "Fully replace the default tool description. "
            "When set, verbose is ignored for the tool description."
        ),
    )
    allow_llm_selection: bool = Field(
        False,
        description=(
            "Let the delegating agent pick the target agent's LLM by name "
            "(within the target's allowed names). Overridable per delegable "
            "agent via DelegableAgent.allow_llm_selection."
        ),
    )


class AgentDelegationProvider:
    def __init__(
        self,
        agent_creator: AgentCreator,
        delegable_agents_sources: List[DelegableAgentsSource],
        enabled: bool,
        verbose: bool,
        include_in_prompt: bool,
        extra_instructions: Optional[str],
        tool_description: Optional[str],
        include_sources: Optional[Set[str]] = None,
        exclude_sources: Optional[Set[str]] = None,
        allow_llm_selection: bool = False,
    ):
        self.__agent_creator = agent_creator
        self.__delegable_agents_sources = delegable_agents_sources
        self.__enabled = enabled
        self.__verbose = verbose
        self.__include_in_prompt = include_in_prompt
        self.__extra_instructions = extra_instructions
        self.__tool_description = tool_description
        self.__include_sources = include_sources
        self.__exclude_sources = exclude_sources
        self.__allow_llm_selection = allow_llm_selection
        self.__resolved: Optional[Dict[str, List[DelegableAgent]]] = None

    async def describe_loaded(self) -> Dict[str, Any]:
        resolved = await self.__resolve_delegable_agents()
        return {
            "enabled": self.__enabled,
            "sources": list(resolved.keys()),
            "agents_by_source": {
                name: [a.name for a in agents] for name, agents in resolved.items()
            },
        }

    async def get_tools(self) -> List[Tool]:
        if not self.__enabled:
            return []

        resolved = await self.__resolve_delegable_agents()
        all_agents = [a for agents in resolved.values() for a in agents]
        if not all_agents:
            return []

        return [self.__build_delegate_tool(all_agents)]

    async def __resolve_delegable_agents(
        self,
    ) -> Dict[str, List[DelegableAgent]]:
        if self.__resolved is not None:
            return self.__resolved
        self.__resolved = {}
        if not self.__enabled:
            return self.__resolved
        for source in self.__active_sources():
            try:
                source_agents = await source.get_agents()
                self.__resolved[source.source_name] = source_agents
            except Exception:
                logger.warning(
                    "Failed to resolve delegable agents",
                    extra={"source": source.source_name},
                )
                self.__resolved[source.source_name] = []
        return self.__resolved

    def __active_sources(self) -> List[DelegableAgentsSource]:
        return [
            source
            for source in self.__delegable_agents_sources
            if is_source_active(
                source.source_name,
                source.enabled,
                self.__include_sources,
                self.__exclude_sources,
            )
        ]

    @staticmethod
    def __delegation_failure_message(agent_name: str) -> str:
        return (
            f"The '{agent_name}' agent could not complete this task. You can try "
            "delegating again, or continue using the information already available."
        )

    def __build_delegate_tool(
        self,
        delegable_agents: List[DelegableAgent],
    ) -> Tool:
        agents_by_name: Dict[str, DelegableAgent] = {
            a.name: a for a in delegable_agents
        }
        identifier_counts = Counter(a.agent_identifier for a in delegable_agents)
        agents_by_alias: Dict[str, DelegableAgent] = dict(agents_by_name)
        for agent in delegable_agents:
            if identifier_counts[agent.agent_identifier] == 1:
                agents_by_alias[agent.agent_identifier] = agent
        agent_names = list(agents_by_name.keys())

        agents_description = "\n".join(
            f"- `{a.name}`: {a.description}" for a in delegable_agents
        )

        def llm_selection_allowed(agent: DelegableAgent) -> bool:
            if agent.allow_llm_selection is not None:
                return agent.allow_llm_selection
            return self.__allow_llm_selection

        any_llm_selection = any(llm_selection_allowed(a) for a in delegable_agents)

        async def execute(
            agent: str,
            prompt: str,
            description: str,
            llm_configuration_name: Optional[str] = None,
        ):
            target = agents_by_alias.get(agent)
            if target is None:
                available = ", ".join(agent_names)
                return f"Unknown agent '{agent}'. Available agents: {available}"

            bot_params = target.bot_params
            if llm_configuration_name and llm_selection_allowed(target):
                bot_params = {
                    **(bot_params or {}),
                    "llm_selection_configuration": {
                        "llm_configuration_name": llm_configuration_name
                    },
                }

            try:
                child = await run_sub_agent(
                    agent_creator=self.__agent_creator,
                    target_identifier=target.agent_identifier,
                    prompt=prompt,
                    bot_params=bot_params,
                )
            except ValueError as exc:
                if str(exc).startswith("Unknown agent:"):
                    return (
                        f"Configured delegate '{target.name}' points to unavailable "
                        f"agent identifier '{target.agent_identifier}'. This is a "
                        "configuration issue: use a registered agent_identifier "
                        "or provide bot_params for a registered base agent."
                    )
                logger.exception(f"Delegation to '{target.name}' failed")
                return self.__delegation_failure_message(target.name)
            except Exception:
                logger.exception(f"Delegation to '{target.name}' failed")
                return self.__delegation_failure_message(target.name)
            return SubAgentToolResult(
                text=child.content or "Sub-agent completed but produced no output.",
                content_parts=child.content_parts or [],
            )

        if self.__tool_description is not None:
            base_description = self.__tool_description
        elif self.__verbose:
            base_description = _VERBOSE_DELEGATE_TOOL_DESCRIPTION
        else:
            base_description = _MINIMAL_DELEGATE_TOOL_DESCRIPTION

        description = f"{base_description}\n\nAvailable agents:\n{agents_description}"

        if self.__extra_instructions:
            description = f"{description}\n\n{self.__extra_instructions}"

        parameters_spec: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": agent_names,
                    "description": "The name of the agent to delegate to.",
                },
                "description": {
                    "type": "string",
                    "description": "A short (3-5 word) label for the task.",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "A detailed, self-contained task description "
                        "for the target agent."
                    ),
                },
            },
            "required": ["agent", "description", "prompt"],
        }
        if any_llm_selection:
            parameters_spec["properties"]["llm_configuration_name"] = {
                "type": "string",
                "description": (
                    "Optional named LLM configuration for the target agent "
                    "(one its setup allows)."
                ),
            }

        return Tool(
            name="delegate",
            description=description,
            executable=execute,
            parameters_spec=parameters_spec,
            streaming_config=ToolStreamingConfig(
                running_text="Delegating to {{ agent }}: {{ description }}...",
                completed_text="Completed delegation to {{ agent }}: {{ description }}",
            ),
        )

    async def to_prompt(self) -> str:
        if not self.__enabled or not self.__include_in_prompt:
            return ""

        sections = []

        delegate_prompt = (
            _VERBOSE_DELEGATE_SYSTEM_PROMPT
            if self.__verbose
            else _MINIMAL_DELEGATE_SYSTEM_PROMPT
        )
        sections.append(delegate_prompt)

        if self.__extra_instructions:
            sections.append(self.__extra_instructions)

        return "\n\n".join(sections)


class AgentDelegationProviderFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        agent_creator: AgentCreator,
        delegable_agents_source_group: DelegableAgentsSourceGroup = (
            DelegableAgentsSourceGroup(items=[])
        ),
        agent_delegation_provider_configuration: AgentDelegationProviderConfiguration = AgentDelegationProviderConfiguration(),  # noqa: E501
    ) -> AgentDelegationProvider:
        return AgentDelegationProvider(
            agent_creator=agent_creator,
            delegable_agents_sources=delegable_agents_source_group.items,
            enabled=agent_delegation_provider_configuration.enabled,
            verbose=agent_delegation_provider_configuration.verbose,
            include_in_prompt=agent_delegation_provider_configuration.include_in_prompt,
            extra_instructions=(
                agent_delegation_provider_configuration.extra_instructions
            ),
            tool_description=agent_delegation_provider_configuration.tool_description,
            include_sources=(
                set(agent_delegation_provider_configuration.include_sources)
                if agent_delegation_provider_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(agent_delegation_provider_configuration.exclude_sources)
                if agent_delegation_provider_configuration.exclude_sources
                else None
            ),
            allow_llm_selection=(
                agent_delegation_provider_configuration.allow_llm_selection
            ),
        )
