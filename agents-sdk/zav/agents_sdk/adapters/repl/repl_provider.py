import inspect
import json
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion
from zav.agents_sdk.adapters.repl.repl_sandbox import (
    HostFunctions,
    ReplSandbox,
    ReplSandboxGroup,
)
from zav.agents_sdk.adapters.tool_confirmation.session_approvals import (
    SessionApprovals,
)
from zav.agents_sdk.adapters.tool_policy import (
    ToolPolicyProvider,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import (
    Tool,
    ToolsRegistry,
    ToolStreamingConfig,
)

RUN_PYTHON_TOOL_NAME = "run_python"

_RUN_PYTHON_DESCRIPTION = """\
Run Python in a persistent sandboxed REPL. Variables, functions and imports \
survive between calls. The other tools are available as async functions with \
the same names and parameters; `await` them and work on their results as \
Python objects. The value of the last expression and anything printed are \
returned.\
"""

_SYSTEM_PROMPT_SECTION = """\
## Python

`run_python` gives you a persistent Python REPL. Use it when a task needs \
loops, filtering, joining or transforming data, or when a tool would otherwise \
return more than you want to read: call tools from Python, keep the results in \
variables, and print only what you need. Variables persist between calls within \
this turn.\
"""

_ASK_FIRST_MESSAGE = (
    "{name} needs the user's approval, which cannot be asked from Python. "
    "Call {name} directly as a tool so the user can approve it."
)

_TOOL_STUBS_HEADER = (
    "Tools callable from Python (all are `async`, call them with `await`):"
)

_JSON_TO_PYTHON_TYPE = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


class ReplProviderConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Add `run_python`: a persistent sandboxed Python REPL in which the "
            "agent's other tools are callable as functions."
        ),
    )
    sandbox: Optional[str] = Field(
        None,
        description="source_name of the sandbox to run in. Default: first enabled.",
    )
    expose_tools: bool = Field(
        True,
        description=(
            "Expose the agent's other tools as Python functions. To make a tool "
            "reachable only from Python, and not as a direct tool call, list it in "
            "the tools provider's `sandbox_only_sources` or `sandbox_only_tools`."
        ),
    )
    max_output_chars: int = Field(
        20_000, description="Truncate a call's combined output beyond this."
    )


_DOCSTRING_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Returns?|Yields?|Raises|Examples?|Notes?):\s*$"
)


def docstring_sections(description: str) -> Dict[str, str]:
    """Split a Google-style docstring into its summary and named sections."""
    sections: Dict[str, List[str]] = {"": []}
    current = ""
    for line in inspect.cleandoc(description).splitlines():
        match = _DOCSTRING_SECTION.match(line.strip())
        if match:
            current = match.group(1).rstrip("s").lower()
            sections[current] = []
            continue
        sections[current].append(line)
    return {
        name: textwrap.dedent("\n".join(lines)).strip()
        for name, lines in sections.items()
    }


def render_tool_stub(tool: Tool) -> str:
    """A Python signature plus the docstring parts the code author needs: the
    summary and what the call returns. Parameter descriptions are included only
    for sandbox-only tools; for the others the model already has them in the
    tool definitions."""
    spec = tool.get_parameters_spec() or {}
    properties: Dict[str, Any] = spec.get("properties", {}) or {}
    required = set(spec.get("required", []) or [])
    defaults = tool.get_parameter_defaults()
    params: List[str] = []
    for name, schema in properties.items():
        python_type = _JSON_TO_PYTHON_TYPE.get((schema or {}).get("type", ""), "Any")
        if name in required:
            params.append(f"{name}: {python_type}")
        else:
            params.append(f"{name}: {python_type} = {defaults.get(name)!r}")
    signature = f"async def {tool.name}({', '.join(params)}) -> Any:"
    sections = docstring_sections(tool.description or "")
    body = [sections.get("", "")]
    if "direct" not in tool.allowed_callers and sections.get("arg"):
        body.append("Args:\n" + textwrap.indent(sections["arg"], "    "))
    if sections.get("return"):
        body.append("Returns:\n" + textwrap.indent(sections["return"], "    "))
    docstring = "\n\n".join(part for part in body if part)
    if not docstring:
        return signature + "\n    ..."
    return signature + "\n" + textwrap.indent(f'"""{docstring}\n"""', "    ")


def positional_to_keyword(tool: Tool, args: tuple, kwargs: Dict[str, Any]):
    spec = tool.get_parameters_spec() or {}
    names = list((spec.get("properties", {}) or {}).keys())
    if len(args) > len(names):
        raise TypeError(
            f"{tool.name}() takes at most {len(names)} positional arguments "
            f"({len(args)} given)"
        )
    params = dict(zip(names, args))
    params.update(kwargs)
    return params


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return json.loads(value.json())
    return json.loads(json.dumps(value, default=str))


class ReplProvider:
    """Code as the orchestration layer: ``run_python`` gets the registry of the
    round it runs in, so Python can call that round's tools as functions, as
    the turn selected them and with their options. A tool that asks first is
    declined, since an approval cannot be asked from Python, unless the user
    allowed it for this chat."""

    def __init__(
        self,
        sandboxes: List[ReplSandbox],
        enabled: bool,
        sandbox_name: Optional[str],
        expose_tools: bool,
        max_output_chars: int,
        tool_policy: ToolPolicyProvider,
        session_approvals: SessionApprovals,
    ) -> None:
        self.__tool_policy = tool_policy
        self.__session_approvals = session_approvals
        self.__sandboxes = sandboxes
        self.__enabled = enabled
        self.__sandbox_name = sandbox_name
        self.__expose_tools = expose_tools
        self.__max_output_chars = max_output_chars

    def describe_loaded(self) -> Dict[str, Any]:
        sandbox = self.__sandbox()
        return {
            "enabled": self.__enabled,
            "sources": [s.source_name for s in self.__active_sandboxes()],
            "sandbox": sandbox.source_name if sandbox else None,
        }

    async def get_tools(self) -> List[Tool]:
        if self.__sandbox() is None:
            return []

        # The round's registry runs this tool and hands itself in.
        async def run_python(code: str, tools_registry: ToolsRegistry) -> str:
            return await self.run_python(code, tools_registry)

        return [
            Tool(
                name=RUN_PYTHON_TOOL_NAME,
                description=_RUN_PYTHON_DESCRIPTION,
                executable=run_python,
                parameters_spec={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source."}
                    },
                    "required": ["code"],
                },
                streaming_config=ToolStreamingConfig(
                    running_text="Running Python...",
                    completed_text="Ran Python",
                ),
            )
        ]

    async def to_prompt(
        self, tools: List[Tool], transcript: List[ChatCompletion]
    ) -> str:
        """Built once per turn, from the tools the turn's first round offers;
        a tool a later round leaves out is refused when Python calls it."""
        sandbox = self.__sandbox()
        if sandbox is None:
            return ""
        sections = [_SYSTEM_PROMPT_SECTION, await sandbox.to_prompt()]
        first_round = await self.__tool_policy.apply(tools, transcript)
        stubs = [
            render_tool_stub(tool)
            for tool, asks_first in self.__exposed_tools(first_round.tools)
            if not asks_first
        ]
        if stubs:
            sections.append(
                "\n\n".join([_TOOL_STUBS_HEADER, "```python", *stubs, "```"])
            )
        return "\n\n".join(section for section in sections if section)

    async def run_python(self, code: str, tools_registry: ToolsRegistry) -> str:
        """Python calls the tools of the round's registry it may call."""
        sandbox = self.__sandbox()
        if sandbox is None:
            return "Error: no Python sandbox is configured."
        output = await sandbox.run(code, self.__host_functions(tools_registry))
        return self.__truncate(output)

    async def cleanup(self) -> None:
        for sandbox in self.__sandboxes:
            try:
                await sandbox.aclose()
            except Exception as e:
                logger.warning(f"Error closing sandbox {sandbox.source_name}: {e}")

    def __exposed_tools(self, tools: List[Tool]) -> List[Tuple[Tool, bool]]:
        """The tools Python may call, each with whether it asks first."""
        if not self.__expose_tools:
            return []
        return [
            (
                tool,
                tool.options.confirmation == "require"
                or (
                    tool.options.confirmation == "ask"
                    and not self.__session_approvals.allows(tool.name)
                ),
            )
            for tool in tools
            if "sandbox" in tool.allowed_callers and tool.name != RUN_PYTHON_TOOL_NAME
        ]

    def __host_functions(self, tools_registry: ToolsRegistry) -> HostFunctions:
        return {
            tool.name: self.__host_function(tools_registry, tool, asks_first)
            for tool, asks_first in self.__exposed_tools(
                list(tools_registry.tools_index.values())
            )
        }

    @staticmethod
    def __host_function(
        tools_registry: ToolsRegistry, tool: Tool, asks_first: bool
    ) -> Callable:
        async def call(*args: Any, **kwargs: Any) -> Any:
            if asks_first:
                raise PermissionError(_ASK_FIRST_MESSAGE.format(name=tool.name))
            params = positional_to_keyword(tool, args, kwargs)
            return json_safe(await tools_registry.execute(tool.name, params))

        call.__name__ = tool.name
        return call

    def __truncate(self, text: str) -> str:
        if len(text) <= self.__max_output_chars:
            return text
        omitted = len(text) - self.__max_output_chars
        return text[: self.__max_output_chars] + f"\n... [{omitted} chars truncated]"

    def __active_sandboxes(self) -> List[ReplSandbox]:
        if not self.__enabled:
            return []
        return [
            sandbox
            for sandbox in self.__sandboxes
            if is_source_active(sandbox.source_name, sandbox.enabled, None, None)
        ]

    def __sandbox(self) -> Optional[ReplSandbox]:
        active = self.__active_sandboxes()
        if not active:
            return None
        if self.__sandbox_name is None:
            return active[0]
        for sandbox in active:
            if sandbox.source_name == self.__sandbox_name:
                return sandbox
        logger.warning(f"Configured sandbox '{self.__sandbox_name}' is not enabled.")
        return None


class ReplProviderFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        tool_policy_provider: ToolPolicyProvider,
        session_approvals: SessionApprovals,
        repl_sandbox_group: ReplSandboxGroup = ReplSandboxGroup(items=[]),
        repl_provider_configuration: ReplProviderConfiguration = (
            ReplProviderConfiguration()
        ),
    ) -> ReplProvider:
        configuration = repl_provider_configuration
        return ReplProvider(
            sandboxes=repl_sandbox_group.items,
            enabled=configuration.enabled,
            sandbox_name=configuration.sandbox,
            expose_tools=configuration.expose_tools,
            max_output_chars=configuration.max_output_chars,
            tool_policy=tool_policy_provider,
            session_approvals=session_approvals,
        )
