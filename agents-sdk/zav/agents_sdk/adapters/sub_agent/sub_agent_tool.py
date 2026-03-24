from typing import Any, Dict, List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.sub_agent.delegable_agents_source import (
    DelegableAgent,
    DelegableAgentsSource,
    DelegableAgentsSourceGroup,
)
from zav.agents_sdk.domain.agent_creator import AgentCreator
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage, ChatMessageSender
from zav.agents_sdk.domain.tools import Tool, ToolStreamingConfig

_MINIMAL_TASK_TOOL_DESCRIPTION = """\
Launch an isolated sub-agent to perform a focused, well-scoped unit of work. \
The sub-agent runs in a fresh context with no access to your conversation \
history. It executes autonomously and returns a single text result.\
"""

_VERBOSE_TASK_TOOL_DESCRIPTION = """\
Launch an isolated sub-agent to perform a focused, well-scoped unit of work. \
The sub-agent runs in a fresh context with no access to your conversation \
history. It executes autonomously and returns a single text result.

CRITICAL — Task granularity:
- Each task should be a SMALL, FOCUSED unit of work — not a restatement of \
your entire problem.
- NEVER delegate your whole task to a single sub-agent. You are the \
orchestrator: you gather data, dispatch sub-agents for per-item work, \
and aggregate results yourself.
- When processing a list of items (commits, files, PRs, etc.), launch one \
task per item or per small batch — not one task for the entire list.

Writing effective prompts:
1. Prompts must be SELF-CONTAINED — include the specific data the sub-agent \
needs (e.g., a commit SHA, a file path, a PR number). It cannot see your \
conversation.
2. State explicitly what tools the sub-agent should use AND what to avoid \
(e.g., "use search to find relevant documents; do not fetch full file contents").
3. Specify exact output format AND length constraints \
(e.g., "one paragraph, ≤90 words" or "bullet list, max 5 items"). \
Keep it simple — avoid JSON unless necessary.
4. If code must be quoted, cap it (e.g., "≤10 lines total"). \
Avoid returning raw diffs or patches unless explicitly requested.
5. The user CANNOT see the sub-agent's output — you must relay the result \
yourself.

Performance:
- Launch multiple task calls in a single response to run them concurrently.
- Trust sub-agent outputs by default; validate only when there are \
contradictions or high-stakes accuracy requirements.
- Sub-agents run in their own context window. Delegating data-heavy work \
(fetching large context, listing many items, reading diffs) keeps those \
payloads out of YOUR context and avoids hitting the context budget.
- Default rule: if the job is "do the same investigation/summarization \
across a list of items", delegate it.

When NOT to use this tool:
- If you can answer with a single tool call (one search, one file read, one command).
- If you already have the data and just need to compute/format the answer.
- If the task depends on nuances from your conversation that are hard to \
summarize.\
"""

_MINIMAL_DELEGATE_TOOL_DESCRIPTION = """\
Delegate a task to a specialized agent. The target agent runs in a fresh context \
with no access to your conversation history. It executes autonomously using its \
own tools and capabilities, and returns a single text result.\
"""

_VERBOSE_DELEGATE_TOOL_DESCRIPTION = """\
Delegate a task to a specialized agent. The target agent runs in a fresh context \
with no access to your conversation history. It executes autonomously using its \
own tools and capabilities, and returns a single text result.

Use this tool when:
- The user's request matches the expertise of a specific agent.
- A specialized agent has tools or knowledge better suited to the task.
- You want to leverage a domain-specific agent for a focused piece of work.

Writing effective prompts:
1. Prompts must be SELF-CONTAINED — include all relevant context the target \
agent needs. It cannot see your conversation.
2. Specify exact output format AND length constraints \
(e.g., "one paragraph, ≤90 words" or "bullet list, max 5 items").
3. The user CANNOT see the delegated agent's output — you must relay the \
result yourself.

When NOT to use this tool:
- If you can handle the task yourself with your own tools.
- If the task depends on nuances from your conversation that are hard to \
summarize.\
"""

_MINIMAL_SYSTEM_PROMPT_SECTION = """\
## Sub-agent delegation

You have access to a `task` tool that launches an isolated sub-agent for focused \
work. The sub-agent runs in a fresh context — it cannot see your conversation \
history — and returns a single text result.\
"""

_VERBOSE_SYSTEM_PROMPT_SECTION = """\
## Sub-agent delegation

You have access to a `task` tool that launches an isolated sub-agent for focused \
work. The sub-agent runs in a fresh context — it cannot see your conversation \
history — and returns a single text result.

You are the ORCHESTRATOR. Sub-agents are WORKERS. Follow this pattern:
1. Gather data yourself (list items, fetch metadata, identify what needs processing).
2. Fan out — launch one `task` per item or small batch, concurrently.
3. Aggregate — collect the results and synthesize the final answer yourself.

Never delegate your entire task to a single sub-agent. Each sub-agent should \
handle one focused piece: analyze one commit, review one file, research one topic.

Default rule: if the job is "do the same investigation or summarization across \
a list of items" (PRs, commits, files, issues), delegate it. This prevents \
large tool outputs from consuming your main context.

When delegating, constrain the sub-agent's output:
- Provide self-contained inputs (repo, IDs, paths, SHAs).
- Specify which tools to use AND what to avoid.
- Define strict output format and length (e.g., "one paragraph, ≤90 words").
- Cap code quoting (e.g., "≤10 lines total"). Avoid raw diffs/patches unless \
explicitly requested.

Prefer the task tool when:
- You have a list of items that each need independent investigation or enrichment.
- Research requires exploring unfamiliar areas that would pollute your context.
- A tool call is likely to return a large output (API responses, file contents, \
logs, diffs). Delegating keeps the payload in the sub-agent's context instead \
of consuming your own context budget.
- Work can be parallelized — launch multiple task calls in a single response.

Do NOT use the task tool when:
- You can answer with a single tool call (one search, one file read).
- You already have the data and just need to compute or format the answer.
- The task depends on nuances from your conversation that are hard to summarize.

Trust sub-agent outputs by default. Validate only when there are contradictions \
or high-stakes accuracy requirements.\
"""

_MINIMAL_DELEGATE_SYSTEM_PROMPT_SECTION = """\
## Agent delegation

You have access to a `delegate` tool that delegates a task to a specialized agent. \
Each agent has its own tools and capabilities. The target agent runs in a fresh \
context — it cannot see your conversation history — and returns a single text result.\
"""

_VERBOSE_DELEGATE_SYSTEM_PROMPT_SECTION = """\
## Agent delegation

You have access to a `delegate` tool that delegates a task to a specialized agent. \
Each agent has its own tools and capabilities. The target agent runs in a fresh \
context — it cannot see your conversation history — and returns a single text result.

Use `delegate` when a specialized agent is better suited for the task — for example, \
when the user explicitly asks for a specific agent's capabilities, or when a domain \
expert agent would produce a better result than your general-purpose tools.

When delegating:
- Write a SELF-CONTAINED prompt that includes all relevant context.
- Specify the desired output format and length constraints.
- The user cannot see the delegated agent's output — relay the result yourself.

Prefer your own tools over delegation when you can handle the task directly.\
"""


class SubAgentConfiguration(BaseModel):
    """Configuration for sub-agent delegation behavior."""

    enabled: bool = Field(
        False,
        description="Enable sub-agent delegation.",
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


class SubAgentTool:
    def __init__(
        self,
        agent_creator: AgentCreator,
        enabled: bool,
        verbose: bool,
        include_in_prompt: bool,
        extra_instructions: Optional[str],
        tool_description: Optional[str],
        delegable_agents_sources: Optional[List[DelegableAgentsSource]] = None,
    ):
        self.__agent_creator = agent_creator
        self.__enabled = enabled
        self.__verbose = verbose
        self.__include_in_prompt = include_in_prompt
        self.__extra_instructions = extra_instructions
        self.__tool_description = tool_description
        self.__delegable_agents_sources = delegable_agents_sources or []

    async def get_tools(self, agent_identifier: str) -> List[Tool]:
        if not self.__enabled:
            return []

        tools = [self.__build_task_tool(agent_identifier)]

        delegable_agents = await self.__resolve_delegable_agents()
        if delegable_agents:
            tools.append(self.__build_delegate_tool(delegable_agents))

        return tools

    async def __resolve_delegable_agents(self) -> List[DelegableAgent]:
        agents: List[DelegableAgent] = []
        for source in self.__delegable_agents_sources:
            try:
                agents.extend(await source.get_agents())
            except Exception:
                logger.warning(
                    "Failed to resolve delegable agents",
                    extra={"source": source.source_name},
                )
        return agents

    def __build_task_tool(self, agent_identifier: str) -> Tool:
        async def execute(description: str, prompt: str) -> str:
            return await self.__run_agent(
                target_identifier=agent_identifier,
                description=description,
                prompt=prompt,
            )

        if self.__tool_description is not None:
            description = self.__tool_description
        elif self.__verbose:
            description = _VERBOSE_TASK_TOOL_DESCRIPTION
        else:
            description = _MINIMAL_TASK_TOOL_DESCRIPTION

        if self.__extra_instructions:
            description = f"{description}\n\n{self.__extra_instructions}"

        return Tool(
            name="task",
            description=description,
            executable=execute,
            parameters_spec={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A short (3-5 word) label for the task.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "A detailed, self-contained task description "
                            "for the sub-agent."
                        ),
                    },
                },
                "required": ["description", "prompt"],
            },
            streaming_config=ToolStreamingConfig(
                running_text="Running task: {{ description }}...",
                completed_text="Completed task: {{ description }}",
            ),
        )

    def __build_delegate_tool(
        self,
        delegable_agents: List[DelegableAgent],
    ) -> Tool:
        agents_by_name: Dict[str, DelegableAgent] = {
            a.name: a for a in delegable_agents
        }
        agent_names = list(agents_by_name.keys())

        agents_description = "\n".join(
            f"- `{a.name}`: {a.description}" for a in delegable_agents
        )

        async def execute(agent: str, description: str, prompt: str) -> str:
            target = agents_by_name.get(agent)
            if target is None:
                available = ", ".join(agent_names)
                return f"Unknown agent '{agent}'. Available agents: {available}"

            return await self.__run_agent(
                target_identifier=target.agent_identifier,
                description=description,
                prompt=prompt,
            )

        if self.__verbose:
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

    async def __run_agent(
        self,
        target_identifier: str,
        description: str,
        prompt: str,
    ) -> str:
        logger.info(
            "Spawning sub-agent",
            extra={
                "agent_identifier": target_identifier,
                "task_description": description,
            },
        )

        agent = await self.__agent_creator.create(
            agent_identifier=target_identifier,
            bot_params={"enable_sub_agents": False},
        )

        conversation = [
            ChatMessage(sender=ChatMessageSender.USER, content=prompt),
        ]

        result = await agent.execute(conversation)

        if result and result.content:
            return result.content

        return "Sub-agent completed but produced no output."

    def to_prompt(self) -> str:
        if not self.__enabled or not self.__include_in_prompt:
            return ""

        sections = []

        task_prompt = (
            _VERBOSE_SYSTEM_PROMPT_SECTION
            if self.__verbose
            else _MINIMAL_SYSTEM_PROMPT_SECTION
        )
        sections.append(task_prompt)

        if self.__delegable_agents_sources:
            delegate_prompt = (
                _VERBOSE_DELEGATE_SYSTEM_PROMPT_SECTION
                if self.__verbose
                else _MINIMAL_DELEGATE_SYSTEM_PROMPT_SECTION
            )
            sections.append(delegate_prompt)

        if self.__extra_instructions:
            sections.append(self.__extra_instructions)

        return "\n\n".join(sections)


class SubAgentToolFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        agent_creator: AgentCreator,
        delegable_agents_source_group: DelegableAgentsSourceGroup = (
            DelegableAgentsSourceGroup(items=[])
        ),
        sub_agent_configuration: SubAgentConfiguration = SubAgentConfiguration(),
    ) -> SubAgentTool:
        return SubAgentTool(
            agent_creator=agent_creator,
            enabled=sub_agent_configuration.enabled,
            verbose=sub_agent_configuration.verbose,
            include_in_prompt=sub_agent_configuration.include_in_prompt,
            extra_instructions=sub_agent_configuration.extra_instructions,
            tool_description=sub_agent_configuration.tool_description,
            delegable_agents_sources=delegable_agents_source_group.items,
        )
