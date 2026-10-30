import uuid
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional

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

    async def check_result(
        self,
        call: ToolCallRequest,
        result: Any,
        transcript: Optional[List[ChatCompletion]],
    ) -> Optional[ResultDecision]:
        replaced = False
        reasons: List[str] = []
        for source in self.__sources:
            decision = await source.check_result(call, result, transcript)
            if decision is None:
                continue
            if decision.action == "block":
                return decision
            result, replaced = decision.value, True
            if decision.reason and decision.reason not in reasons:
                reasons.append(decision.reason)
        return ResultDecision.replace(result, " ".join(reasons)) if replaced else None

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
        result = await tools_registry.execute(name=name, params=params)
        checked = await self.check_result(call, result, None)
        if checked is not None and checked.action == "block":
            raise ToolCallRefused(checked.message)
        return checked.value if checked is not None else result

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
