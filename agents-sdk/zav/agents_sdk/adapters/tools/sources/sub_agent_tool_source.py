from typing import List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_creator import AgentCreator, run_sub_agent
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
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

_MINIMAL_TASK_SYSTEM_PROMPT = """\
## Sub-agent delegation

You have access to a `task` tool that launches an isolated sub-agent for focused \
work. The sub-agent runs in a fresh context — it cannot see your conversation \
history — and returns a single text result.\
"""

_VERBOSE_TASK_SYSTEM_PROMPT = """\
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


class SubAgentToolSourceConfiguration(BaseModel):
    """Configuration for the task tool (self-delegation)."""

    enabled: bool = Field(
        True,
        description="Enable sub-agent task tool.",
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


class SubAgentToolSource(ToolsSource):
    """Produces the ``task`` tool that spawns a copy of the current agent."""

    source_name = "sub_agent"

    def __init__(
        self,
        agent_creator: AgentCreator,
        enabled: bool,
        verbose: bool,
        include_in_prompt: bool,
        extra_instructions: Optional[str],
        tool_description: Optional[str],
    ):
        self.__agent_creator = agent_creator
        self.enabled = enabled
        self.__verbose = verbose
        self.__include_in_prompt = include_in_prompt
        self.__extra_instructions = extra_instructions
        self.__tool_description = tool_description

    async def get_tools(self) -> List[Tool]:
        if not self.enabled:
            return []
        return [self.__build_task_tool()]

    def __build_task_tool(self) -> Tool:
        async def execute(prompt: str, description: str) -> str:
            return await run_sub_agent(
                agent_creator=self.__agent_creator,
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

    async def to_prompt(self) -> str:
        if not self.enabled or not self.__include_in_prompt:
            return ""

        sections = []

        task_prompt = (
            _VERBOSE_TASK_SYSTEM_PROMPT
            if self.__verbose
            else _MINIMAL_TASK_SYSTEM_PROMPT
        )
        sections.append(task_prompt)

        if self.__extra_instructions:
            sections.append(self.__extra_instructions)

        return "\n\n".join(sections)


class SubAgentToolSourceFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        agent_creator: AgentCreator,
        sub_agent_tool_source_configuration: SubAgentToolSourceConfiguration = (
            SubAgentToolSourceConfiguration()
        ),
    ) -> SubAgentToolSource:
        return SubAgentToolSource(
            agent_creator=agent_creator,
            enabled=sub_agent_tool_source_configuration.enabled,
            verbose=sub_agent_tool_source_configuration.verbose,
            include_in_prompt=sub_agent_tool_source_configuration.include_in_prompt,
            extra_instructions=sub_agent_tool_source_configuration.extra_instructions,
            tool_description=sub_agent_tool_source_configuration.tool_description,
        )
