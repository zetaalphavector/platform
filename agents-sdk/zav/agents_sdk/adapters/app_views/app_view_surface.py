import asyncio
import inspect
from typing import Any, Dict, List, Optional, Tuple

from zav.logging import logger

from zav.agents_sdk.adapters.mcp.tools_provider import MCPToolsProvider
from zav.agents_sdk.adapters.tool_policy import ToolPolicyProvider
from zav.agents_sdk.adapters.tools.tools_provider import ToolsProvider
from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyFactory,
    AgentDependencyRegistry,
)
from zav.agents_sdk.domain.tools import AppView, Tool, ToolCallRefused, ToolsRegistry

_RULE_ASKS_MESSAGE = (
    "A rule asks the user before this {name} call, which a view cannot do."
)


class AppViewNotFound(LookupError):
    """The agent has no tool by that name with a view, or the view has no
    tool by that name it may call."""


class AppViewSurface:
    """An agent's tools as the views the chat renders reach them: the view a
    tool names, and the tools that view may call. A view calls the tools of
    its own tool's origin (its MCP server or tools source) that admit the
    ``app`` caller, by the name they have there. The tool rules for the
    ``app`` caller apply; a tool or a call that asks first is refused, since
    nobody can be asked."""

    def __init__(
        self,
        tools_provider: ToolsProvider,
        mcp_provider: MCPToolsProvider,
        tool_policy: ToolPolicyProvider,
    ):
        self.__tools_provider = tools_provider
        self.__mcp_provider = mcp_provider
        self.__tool_policy = tool_policy

    async def view(self, view_tool: str) -> Optional[AppView]:
        """The view of the agent's tool ``view_tool``, from the tool's MCP
        server or tools source. None when the view ships with the client."""
        tool, view = await self.__view_tool(view_tool)
        if tool.origin is not None and tool.origin.provider == "mcp":
            return await self.__mcp_provider.read_view(tool.origin.source, view)
        if tool.origin is not None and tool.origin.provider == "tools":
            return await self.__tools_provider.app_view(
                tool.origin.source, tool.origin.name
            )
        return None

    async def call(
        self, view_tool: str, tool_name: str, params: Optional[Dict[str, Any]]
    ) -> Any:
        origin = (await self.__view_tool(view_tool))[0].origin
        callable_tools = [
            tool
            for tool in await self.__tools()
            if "app" in tool.allowed_callers
            and tool.origin is not None
            and origin is not None
            and (tool.origin.provider, tool.origin.source)
            == (origin.provider, origin.source)
        ]
        tool = next(
            (t for t in callable_tools if t.origin and t.origin.name == tool_name),
            None,
        )
        if tool is None:
            raise AppViewNotFound(f"The view cannot call a tool named '{tool_name}'.")
        if tool.options.confirmation in ("ask", "require"):
            raise ToolCallRefused(_RULE_ASKS_MESSAGE.format(name=tool.name))
        self.__check_arguments(tool, params or {})
        tools_registry = ToolsRegistry()
        tools_registry.extend(callable_tools)
        try:
            return await self.__tool_policy.for_caller("app").execute_outside_turn(
                tools_registry,
                tool.name,
                params,
                _RULE_ASKS_MESSAGE.format(name=tool.name),
            )
        except ToolCallRefused:
            raise
        except Exception:
            # The view gets an MCP tool error without the tool's internals.
            logger.exception(f"The view call of {tool.name} failed")
            return {
                "content": [{"type": "text", "text": f"The {tool_name} call failed."}],
                "isError": True,
            }

    async def cleanup(self) -> None:
        await asyncio.gather(
            self.__tools_provider.cleanup(), self.__mcp_provider.cleanup()
        )

    @staticmethod
    def __check_arguments(tool: Tool, params: Dict[str, Any]) -> None:
        """Raises ValueError when ``params`` do not bind to the tool's signature.
        Value types stay the tool's own check, as for a model's call."""
        signature = inspect.signature(tool.executable)
        # The registry running the tool passes itself to these parameters.
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.annotation not in (ToolsRegistry, "ToolsRegistry")
        ]
        try:
            signature.replace(parameters=parameters).bind(**params)
        except TypeError as e:
            raise ValueError(f"The {tool.name} call has invalid arguments: {e}")

    async def __tools(self) -> List[Tool]:
        collected = await asyncio.gather(
            self.__tools_provider.get_tools(), self.__mcp_provider.get_tools()
        )
        offered = await self.__tool_policy.apply(
            [tool for tools in collected for tool in tools], None
        )
        return offered.tools

    async def __view_tool(self, name: str) -> Tuple[Tool, str]:
        tool = next((t for t in await self.__tools() if t.name == name), None)
        view = tool.display.view if tool and tool.display else None
        if tool is None or view is None:
            raise AppViewNotFound(f"The agent has no tool '{name}' with a view.")
        return tool, view


class AppViewSurfaceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        tools_provider: ToolsProvider,
        mcp_provider: MCPToolsProvider,
        tool_policy_provider: ToolPolicyProvider,
    ) -> AppViewSurface:
        return AppViewSurface(
            tools_provider=tools_provider,
            mcp_provider=mcp_provider,
            tool_policy=tool_policy_provider,
        )


AgentDependencyRegistry.register(AppViewSurfaceFactory)
