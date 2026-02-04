from typing import AsyncGenerator

from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
    ZAVChatCompletionClient,
)
from zav.agents_sdk.adapters.mcp.tools_provider import MCPToolsProvider
from zav.agents_sdk.adapters.skills.skills_provider import SkillsProvider
from zav.agents_sdk.domain.chat_agent import StreamableChatAgent
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry
from zav.agents_sdk.domain.chat_message import ChatMessage

_MAIN_SYSTEM_PROMPT = """You are a helpful AI assistant with access to specialized tools and skills.

SKILLS
- You have access to skills that provide detailed instructions for specialized tasks.
- When a task matches an available skill, use the `use_skill` tool to load it.
- Follow the skill instructions carefully once loaded.

TOOLS
- You have access to external tools via MCP (Model Context Protocol) servers.
- These tools can interact with external services like GitHub, databases, APIs, etc.
- Use the appropriate tool when the user's request requires external actions or data.

GENERAL BEHAVIOR
- Be accurate, clear, and efficient.
- Adapt the level of technical detail to the user's apparent expertise.
- Prefer concrete, actionable answers over vague advice."""  # noqa: E501


@ChatAgentClassRegistry.register()
class Agent(StreamableChatAgent):
    agent_name = "agent"

    def __init__(
        self,
        client: ZAVChatCompletionClient,
        skills_provider: SkillsProvider,
        mcp_provider: MCPToolsProvider,
        system_prompt: str = _MAIN_SYSTEM_PROMPT,
    ):
        self.client = client
        self.skills_provider = skills_provider
        self.mcp_provider = mcp_provider
        self.__system_prompt = system_prompt

    async def execute_streaming(
        self, conversation: list[ChatMessage]
    ) -> AsyncGenerator[ChatMessage, None]:
        self.tools_registry.extend(await self.mcp_provider.get_tools())
        self.tools_registry.extend(await self.skills_provider.get_tools())

        skills_prompt = await self.skills_provider.to_prompt()
        if skills_prompt:
            self.__system_prompt = f"{self.__system_prompt}\n\n{skills_prompt}"

        response = await self.client.complete(
            bot_setup_description=self.__system_prompt,
            messages=conversation,
            tools=self.tools_registry,
            stream=True,
            execute_tools=True,
            log_fn=self.debug,
        )
        async for chat_client_response in response:
            if chat_client_response.error is not None:
                raise chat_client_response.error
            if chat_client_response.chat_completion is None:
                raise Exception("No response from chat completion client")

            yield ChatMessage.from_orm(chat_client_response.chat_completion)
