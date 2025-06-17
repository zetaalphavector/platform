from typing import Any, Awaitable, Callable, Dict, List, Optional

from zav.agents_sdk.domain.chat_agent import ChatAgent
from zav.agents_sdk.domain.chat_request import ConversationContext
from zav.agents_sdk.domain.tools import Tool


class AgentCreator:
    def __init__(
        self,
        agent_factory: Callable[
            [
                str,
                Dict[str, Any],
                Optional[ConversationContext],
                Optional[Dict[str, Any]],
            ],
            Awaitable[ChatAgent],
        ],
    ):
        self.__agent_factory = agent_factory

    async def create(
        self,
        agent_identifier: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        bot_params: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[ConversationContext] = None,
    ) -> ChatAgent:
        extra_agent_kwargs = {}
        if system_prompt:
            extra_agent_kwargs["system_prompt"] = system_prompt
        new_agent = await self.__agent_factory(
            agent_identifier,
            bot_params or {},
            conversation_context,
            extra_agent_kwargs if extra_agent_kwargs else None,
        )
        if tools:
            new_agent.tools_registry.tools_index = {
                tool.name: tool for tool in tools or []
            }

        return new_agent
