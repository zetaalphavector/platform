from typing import Dict, List, Optional, Union

from zav.prompt_completion import BotConversation, ChatClientRequest
from zav.prompt_completion import ChatMessage as PcChatMessage

from zav.agents_sdk.adapters.llm_models.chat_turn_request import ChatTurnRequest
from zav.agents_sdk.adapters.llm_models.context_window_manager import (
    ContextWindowManager,
)
from zav.agents_sdk.domain.tools import Tool, ToolsRegistry


def tool_to_chat_client_spec(tool: Tool) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.get_parameters_spec(),
        },
    }


class ChatRequestBuilder:
    """Builds prompt-completion requests from turn state and options."""

    def __init__(self, context_window_manager: ContextWindowManager) -> None:
        self.__context_window_manager = context_window_manager

    async def build(
        self,
        request: ChatTurnRequest,
        pc_messages: List[PcChatMessage],
        tools: Optional[Union[ToolsRegistry, List[Dict]]],
        tool_choice: Optional[str],
        include_functions: bool,
    ) -> ChatClientRequest:
        tools_dict = self.__build_tools_dict(tools)

        if self.__context_window_manager.needs_compaction(
            pc_messages, request.bot_setup_description
        ):
            pc_messages = await self.__context_window_manager.compact(
                pc_messages, request.bot_setup_description
            )

        retrieval_tools = self.__context_window_manager.get_tools()
        if retrieval_tools and isinstance(tools, ToolsRegistry):
            new_tools = [
                tool for tool in retrieval_tools if tool.name not in tools.tools_index
            ]
            if new_tools:
                tools.extend(new_tools)
                if tools_dict is not None:
                    tools_dict.extend(
                        tool_to_chat_client_spec(tool) for tool in new_tools
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
        self, tools: Optional[Union[ToolsRegistry, List[Dict]]]
    ) -> Optional[List[Dict]]:
        if isinstance(tools, ToolsRegistry):
            return [
                tool_to_chat_client_spec(tool) for tool in tools.tools_index.values()
            ]
        return tools
