from typing import AbstractSet, Dict, List, Optional, Union

from zav.llm_domain import ToolSearchMethodSettings
from zav.prompt_completion import BotConversation, ChatClientRequest
from zav.prompt_completion import ChatMessage as PcChatMessage

from zav.agents_sdk.adapters.llm_models.chat_turn_request import ChatTurnRequest
from zav.agents_sdk.adapters.llm_models.context_window_manager import (
    ContextWindowManager,
)
from zav.agents_sdk.domain.tools import Tool, ToolsRegistry


def tool_to_chat_client_spec(tool: Tool, on_demand: bool = False) -> Dict:
    spec: Dict = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.get_parameters_spec(),
        },
    }
    # Tools load on demand by group, as the vendors defer them: a tool without
    # one (a skills tool, say) is always loaded.
    if on_demand and tool.namespace:
        spec["defer_loading"] = True
        spec["namespace"] = {
            "name": tool.namespace.name,
            "description": tool.namespace.description,
        }
    return spec


class ChatRequestBuilder:
    """Builds prompt-completion requests from turn state and options."""

    def __init__(
        self,
        context_window_manager: ContextWindowManager,
        tool_search: Optional[List[ToolSearchMethodSettings]] = None,
    ) -> None:
        self.__context_window_manager = context_window_manager
        self.__tool_search = tool_search or []

    def retrieval_tools(self) -> List[Tool]:
        """The tools that read back tool output compaction cut."""
        return self.__context_window_manager.get_tools()

    async def build(
        self,
        request: ChatTurnRequest,
        pc_messages: List[PcChatMessage],
        tools: Optional[Union[ToolsRegistry, List[Dict]]],
        tool_choice: Optional[str],
        include_functions: bool,
        on_demand: AbstractSet[str] = frozenset(),
    ) -> ChatClientRequest:
        """`on_demand`: the tools the tool policy loads on demand this round;
        without a tool search method they load up front."""
        tools_dict = self.__build_tools_dict(
            tools, on_demand if self.__tool_search else frozenset()
        )

        if self.__context_window_manager.needs_compaction(
            pc_messages, request.bot_setup_description
        ):
            pc_messages = await self.__context_window_manager.compact(
                pc_messages, request.bot_setup_description
            )

        return ChatClientRequest(
            conversation=BotConversation(
                bot_setup_description=request.bot_setup_description,
                messages=pc_messages,
            ),
            max_tokens=request.max_tokens,
            **(
                {"functions": request.functions}
                if include_functions and request.functions is not None
                else {}
            ),
            **({"tools": tools_dict} if tools_dict is not None else {}),
            **(
                {"tool_search": self.__tool_search}
                if self.__tool_search
                and any("defer_loading" in tool for tool in tools_dict or [])
                else {}
            ),
            **({"tool_choice": tool_choice} if tool_choice is not None else {}),
            **({"logprobs": request.logprobs} if request.logprobs is not None else {}),
            **({"seed": request.seed} if request.seed is not None else {}),
            **(
                {"parallel_tool_calls": request.parallel_tool_calls}
                if request.parallel_tool_calls is not None
                else {}
            ),
            **(
                {"reasoning_effort": request.reasoning_effort}
                if request.reasoning_effort is not None
                else {}
            ),
            **(
                {"verbosity": request.verbosity}
                if request.verbosity is not None
                else {}
            ),
        )

    def __build_tools_dict(
        self,
        tools: Optional[Union[ToolsRegistry, List[Dict]]],
        on_demand: AbstractSet[str],
    ) -> Optional[List[Dict]]:
        if isinstance(tools, ToolsRegistry):
            return [
                tool_to_chat_client_spec(tool, tool.name in on_demand)
                for tool in tools.tools_index.values()
            ]
        return tools
