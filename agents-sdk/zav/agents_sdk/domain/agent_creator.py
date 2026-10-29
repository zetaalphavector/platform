from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.chat_agent import ChatAgent
from zav.agents_sdk.domain.chat_message import (
    ChatMessage,
    ChatMessageSender,
    ContentPart,
)
from zav.agents_sdk.domain.chat_request import ConversationContext
from zav.agents_sdk.domain.tools import Tool

# What a sub-agent does with a call that needs the user's approval: ask the
# user through the turn's inbox, decline it, or run it.
SubAgentApprovals = Literal["ask", "decline", "run"]

_APPROVALS_STRICTNESS = {"run": 0, "ask": 1, "decline": 2}


def stricter_approvals(*settings: SubAgentApprovals) -> SubAgentApprovals:
    """A delegation setting may tighten what the delegating agent allows, never
    loosen it."""
    return max(settings, key=_APPROVALS_STRICTNESS.__getitem__)


class SubAgentToolResult(BaseModel):
    text: str
    content_parts: List[ContentPart] = Field(default_factory=list)


class SubAgentCount:
    """Agents started with ``task`` and ``delegate`` in one turn. Every agent of
    the turn's tree holds the same instance, so the count spans all depths."""

    def __init__(self) -> None:
        self.started = 0


class AgentCreator:
    def __init__(
        self,
        agent_factory: Callable[
            [
                str,
                Dict[str, Any],
                Optional[ConversationContext],
                Optional[Dict[str, Any]],
                SubAgentApprovals,
            ],
            Awaitable[ChatAgent],
        ],
        default_agent_identifier: str,
    ):
        self.__agent_factory = agent_factory
        self.default_agent_identifier = default_agent_identifier

    async def create(
        self,
        agent_identifier: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        bot_params: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[ConversationContext] = None,
        sub_agent_approvals: SubAgentApprovals = "ask",
    ) -> ChatAgent:
        extra_agent_kwargs = {}
        if system_prompt:
            extra_agent_kwargs["system_prompt"] = system_prompt
        new_agent = await self.__agent_factory(
            agent_identifier or self.default_agent_identifier,
            bot_params or {},
            conversation_context,
            extra_agent_kwargs if extra_agent_kwargs else None,
            sub_agent_approvals,
        )
        if tools:
            new_agent.tools_registry.tools_index = {
                tool.name: tool for tool in tools or []
            }

        return new_agent


async def run_sub_agent(
    agent_creator: AgentCreator,
    prompt: str,
    target_identifier: Optional[str] = None,
    bot_params: Optional[Dict[str, Any]] = None,
    sub_agent_approvals: SubAgentApprovals = "ask",
    conversation_context: Optional[ConversationContext] = None,
) -> ChatMessage:
    agent = await agent_creator.create(
        agent_identifier=target_identifier,
        bot_params=bot_params,
        conversation_context=conversation_context,
        sub_agent_approvals=sub_agent_approvals,
    )

    conversation = [
        ChatMessage(sender=ChatMessageSender.USER, content=prompt),
    ]

    result = await agent.execute(conversation)

    if result is not None:
        return result

    return ChatMessage(
        sender=ChatMessageSender.BOT,
        content="Sub-agent completed but produced no output.",
    )
