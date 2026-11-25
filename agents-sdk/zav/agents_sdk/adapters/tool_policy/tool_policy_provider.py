import uuid
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional, Tuple

from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ToolCallRequest,
)
from zav.agents_sdk.adapters.tool_policy.tool_policy_source import (
    ToolPolicyConfiguration,
    ToolPolicySource,
    ToolPolicySourceGroup,
    active_rule_sources,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import FunctionCallRequest
from zav.agents_sdk.domain.tools import (
    CallDecision,
    ResultDecision,
    Tool,
    ToolCaller,
    ToolCallFrame,
    ToolCallRefused,
    ToolsRegistry,
)


@dataclass(frozen=True)
class ToolSelection:
    tools: List[Tool]
    # The source that removed each tool that was removed.
    removed_by: Dict[str, str]


class ToolPolicyProvider:
    """Runs the rule sources one after the other, each on what the one before
    it kept and set. No source waives a required ask."""

    def __init__(self, sources: List[ToolPolicySource]):
        self.__sources = sources

    @property
    def sources(self) -> List[ToolPolicySource]:
        """The sources that run, in run order."""
        return list(self.__sources)

    def for_caller(self, caller: ToolCaller) -> "ToolPolicyProvider":
        """The rules that check this caller's calls."""
        return ToolPolicyProvider(
            [source for source in self.__sources if caller in source.callers]
        )

    async def apply(
        self,
        tools: List[Tool],
        transcript: Optional[List[ChatCompletion]],
        excluding: Collection[str] = (),
    ) -> ToolSelection:
        removed_by: Dict[str, str] = {}
        for source in self.__sources:
            if source.source_name in excluding:
                continue
            kept = {tool.name: tool for tool in await source.apply(tools, transcript)}
            for tool in tools:
                if tool.name not in kept:
                    removed_by[tool.name] = source.source_name
            tools = [
                self.__keep_required(tool, kept[tool.name])
                for tool in tools
                if tool.name in kept
            ]
        return ToolSelection(tools=tools, removed_by=removed_by)

    async def check_call(
        self, call: ToolCallRequest, transcript: Optional[List[ChatCompletion]]
    ) -> Optional[CallDecision]:
        asked: Optional[CallDecision] = None
        for source in self.__sources:
            decision = await source.check_call(call, transcript)
            if decision is not None and decision.action == "decline":
                return decision
            asked = asked or decision
        return asked

    async def prepare_call(
        self, call: ToolCallRequest, tool: Tool
    ) -> Optional[Dict[str, Any]]:
        """The params the tool runs with, each source working on what the one
        before it returned; None when no source changed them."""
        prepared: Optional[Dict[str, Any]] = None
        for source in self.__sources:
            params = await source.prepare_call(call, tool)
            if params is None:
                continue
            prepared = params
            call = ToolCallRequest(
                id=call.id,
                function_call_request=FunctionCallRequest(
                    name=call.function_call_request.name, params=params
                ),
            )
        return prepared

    async def check_result(
        self,
        call: ToolCallRequest,
        tool: Tool,
        result: Any,
        transcript: Optional[List[ChatCompletion]],
    ) -> Optional[ResultDecision]:
        changed = edited = False
        reasons: List[str] = []
        for source in self.__sources:
            decision = await source.check_result(call, tool, result, transcript)
            if decision is None:
                continue
            if decision.action == "block":
                return decision
            result, changed = decision.value, True
            edited = edited or decision.action == "replace"
            if decision.reason and decision.reason not in reasons:
                reasons.append(decision.reason)
        if not changed:
            return None
        if edited:
            return ResultDecision.replace(result, " ".join(reasons))
        return ResultDecision.transform(result, " ".join(reasons))

    async def run(
        self,
        tools_registry: ToolsRegistry,
        call: ToolCallRequest,
        tool: Tool,
        transcript: Optional[List[ChatCompletion]],
        frame: Optional[ToolCallFrame] = None,
    ) -> Tuple[Any, Optional[ResultDecision]]:
        """Runs a call the rules let through, from a turn or from Python: with
        the params they prepare, which the registry parses like the call's own,
        then has them check the result. Returns the tool's result and their
        decision on it."""
        prepared = await self.prepare_call(call, tool)
        func = call.function_call_request
        result = await tools_registry.execute(
            name=func.name,
            params=func.params if prepared is None else prepared,
            frame=frame,
        )
        return result, await self.check_result(call, tool, result, transcript)

    async def execute_outside_turn(
        self,
        tools_registry: ToolsRegistry,
        name: str,
        params: Optional[Dict[str, Any]],
        ask_message: str,
    ) -> Any:
        """Runs a call made outside a turn (Python in the REPL, an MCP client):
        there is no transcript and nobody to ask, so a rule's decline, ask
        (``ask_message``) or block raises ``ToolCallRefused``, a
        ``PermissionError`` the REPL's Python code can catch."""
        call = ToolCallRequest(
            id=f"call_{uuid.uuid4().hex}",
            function_call_request=FunctionCallRequest(name=name, params=params),
        )
        decision = await self.check_call(call, None)
        if decision is not None:
            raise ToolCallRefused(
                decision.message if decision.action == "decline" else ask_message
            )
        tool = tools_registry.tools_index.get(name)
        if tool is None:
            return await tools_registry.execute(name=name, params=params)
        # The call's card shows its own params and the result the rules kept.
        frame = tools_registry.child_frame()
        kept = self.__kept_result(tools_registry, call, tool, frame)
        if frame is None:
            return await kept
        return await frame.stream(frame.tool_call_id, tool, params, kept)

    async def __kept_result(
        self,
        tools_registry: ToolsRegistry,
        call: ToolCallRequest,
        tool: Tool,
        frame: Optional[ToolCallFrame],
    ) -> Any:
        result, decision = await self.run(tools_registry, call, tool, None, frame)
        if decision is None:
            return result
        if decision.action == "block":
            raise ToolCallRefused(decision.message)
        return decision.value

    @staticmethod
    def __keep_required(before: Tool, after: Tool) -> Tool:
        if (
            before.options.confirmation != "require"
            or after.options.confirmation == "require"
        ):
            return after
        return after.copy(
            update={
                "options": after.options.copy(
                    update={
                        "confirmation": "require",
                        "confirmation_source": before.options.confirmation_source,
                    }
                )
            }
        )


class ToolPolicyProviderFactory(AgentDependencyFactory):
    # One per agent: its turn, its REPL and its MCP exposure apply alike.
    __singleton__ = True

    @classmethod
    def create(
        cls,
        tool_policy_source_group: ToolPolicySourceGroup = ToolPolicySourceGroup(
            items=[]
        ),
        tool_policy_configuration: ToolPolicyConfiguration = (
            ToolPolicyConfiguration()
        ),
    ) -> ToolPolicyProvider:
        return ToolPolicyProvider(
            active_rule_sources(
                tool_policy_source_group.items, tool_policy_configuration
            )
        )
