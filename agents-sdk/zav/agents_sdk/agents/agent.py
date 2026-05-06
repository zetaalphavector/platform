from typing import AsyncGenerator, Optional, Tuple

from zav.agents_sdk.adapters.agent_delegation.agent_delegation_provider import (
    AgentDelegationProvider,
)
from zav.agents_sdk.adapters.context.context_provider import ContextProvider
from zav.agents_sdk.adapters.dispatch import DispatchProvider
from zav.agents_sdk.adapters.instructions.instructions_provider import (
    InstructionsProvider,
)
from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
    ChatResponse,
    ZAVChatCompletionClient,
)
from zav.agents_sdk.adapters.mcp.tools_provider import MCPToolsProvider
from zav.agents_sdk.adapters.memory.memory_provider import MemoryProvider
from zav.agents_sdk.adapters.message_processing.message_processing_provider import (
    MessageProcessingProvider,
)
from zav.agents_sdk.adapters.skills.skills_provider import SkillsProvider
from zav.agents_sdk.adapters.tools.tools_provider import ToolsProvider
from zav.agents_sdk.domain.chat_agent import StreamableChatAgent
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry
from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext

_MAIN_SYSTEM_PROMPT = """You are a helpful AI assistant with access to specialized tools and skills. You are expected to be precise, safe, and helpful.

# How you work

## Personality

Your default personality and tone is concise, direct, and friendly. You communicate efficiently, always keeping the user clearly informed about ongoing actions without unnecessary detail. You always prioritize actionable guidance, clearly stating assumptions, prerequisites, and next steps. Unless explicitly asked, you avoid excessively verbose explanations about your work.

## Skills

You have access to skills that provide detailed instructions for specialized tasks.

- When a task matches an available skill, use the `use_skill` tool to load it.
- Follow the loaded skill instructions carefully — they take precedence over general behavior for that task.
- Skills are your primary mechanism for domain-specific workflows. Prefer using a skill over improvising when one is available.

## Tools

You have access to external tools via MCP (Model Context Protocol) servers. These tools can interact with external services like GitHub, databases, APIs, and other systems.

- Use the appropriate tool when the user's request requires external actions or data retrieval.
- When you plan to use multiple tools whose inputs don't depend on each other, call them in parallel whenever possible to minimize latency.
- Before invoking tools, briefly inform the user what you're about to do — keep it to one concise sentence that connects prior context to the next action.

## Task execution

Keep going until the user's query is completely resolved before ending your turn. Only stop when you are sure the task is done. Autonomously resolve the query to the best of your ability using the tools and skills available to you. Do NOT guess or make up an answer.

You MUST adhere to the following:

- If a tool call fails, analyze the error and retry with corrected inputs when possible. Only surface the error to the user if you cannot resolve it.
- When a task requires multiple steps, work through them sequentially and keep the user informed of progress at reasonable intervals with concise updates (no more than 1-2 sentences).
- Do not ask the user for information you can obtain via tools.

## Ambition vs. precision

For open-ended or exploratory tasks, feel free to be ambitious — demonstrate creativity and provide thorough, well-structured answers.

For specific, well-scoped requests, be surgical and precise. Answer exactly what was asked without overstepping. Balance being proactive and helpful with respecting the user's intent.

Use judicious initiative: show good judgment about when to add high-value extras versus when to stay tightly focused.

## Presenting your work

Your responses should read naturally, like an update from a concise teammate.

- For casual conversation, brainstorming, or quick questions, respond in a friendly conversational tone.
- For substantive results, use light structure (headers, bullets) to aid scanning — but only when it genuinely helps clarity.
- Brevity is very important as a default. Be concise, but relax this for tasks where additional detail is important for the user's understanding.
- Adapt the level of technical detail to the user's apparent expertise.
- Prefer concrete, actionable answers over vague advice.
- If there's a logical next step you can help with, concisely suggest it.

### Formatting guidelines

- Use section headers only when they improve clarity. Keep them short (1-3 words).
- Use `-` for bullet lists. Group into short lists (4-6 bullets) ordered by importance.
- Wrap commands, identifiers, file paths, and env vars in backticks.
- Match structure to complexity: multi-part results get headers and grouped bullets; simple results get a short list or paragraph.
- Keep tone collaborative and natural. Be concise and factual — no filler.
- Do not nest bullets or create deep hierarchies."""  # noqa: E501


@ChatAgentClassRegistry.register()
class Agent(StreamableChatAgent):
    agent_name = "agent"

    def __init__(
        self,
        client: ZAVChatCompletionClient,
        skills_provider: SkillsProvider,
        mcp_provider: MCPToolsProvider,
        agent_delegation_provider: AgentDelegationProvider,
        tools_provider: ToolsProvider,
        memory_provider: MemoryProvider,
        context_provider: ContextProvider,
        message_processing_provider: MessageProcessingProvider,
        dispatch_provider: DispatchProvider,
        instructions_provider: InstructionsProvider,
        conversation_context: Optional[ConversationContext] = None,
        system_prompt: str = _MAIN_SYSTEM_PROMPT,
    ):
        self.__client = client
        self.__skills_provider = skills_provider
        self.__mcp_provider = mcp_provider
        self.__agent_delegation_provider = agent_delegation_provider
        self.__tools_provider = tools_provider
        self.__memory_provider = memory_provider
        self.__context_provider = context_provider
        self.__message_processing_provider = message_processing_provider
        self.__dispatch_provider = dispatch_provider
        self.__instructions_provider = instructions_provider
        self.__conversation_context = conversation_context
        self.__system_prompt = system_prompt

    async def execute_streaming(
        self, conversation: list[ChatMessage]
    ) -> AsyncGenerator[ChatMessage, None]:
        if self.__dispatch_provider:
            dispatch = await self.__dispatch_provider.try_dispatch(
                conversation=conversation,
                conversation_context=self.__conversation_context,
                emit=self.emit,
            )
            if dispatch is not None:
                async for msg in dispatch:
                    yield msg
                return

        self.tools_registry.extend(await self.__mcp_provider.get_tools())
        self.tools_registry.extend(await self.__skills_provider.get_tools())
        self.tools_registry.extend(await self.__agent_delegation_provider.get_tools())
        self.tools_registry.extend(await self.__tools_provider.get_tools())
        self.tools_registry.extend(await self.__memory_provider.get_tools())
        self.tools_registry.extend(await self.__context_provider.get_tools())

        skills_prompt = await self.__skills_provider.to_prompt()
        if skills_prompt:
            self.__system_prompt = f"{self.__system_prompt}\n\n{skills_prompt}"

        delegate_prompt = self.__agent_delegation_provider.to_prompt()
        if delegate_prompt:
            self.__system_prompt = f"{self.__system_prompt}\n\n{delegate_prompt}"

        tools_prompt = await self.__tools_provider.to_prompt()
        if tools_prompt:
            self.__system_prompt = f"{self.__system_prompt}\n\n{tools_prompt}"

        instructions_prompt = await self.__instructions_provider.to_prompt()
        if instructions_prompt:
            self.__system_prompt = f"{self.__system_prompt}\n\n{instructions_prompt}"

        memory_prompt = await self.__memory_provider.to_prompt()
        if memory_prompt:
            self.__system_prompt = f"{self.__system_prompt}\n\n{memory_prompt}"

        context_prompt = await self.__context_provider.to_prompt(
            initial_context=self.__conversation_context,
        )
        if context_prompt:
            self.__system_prompt = f"{self.__system_prompt}\n\n{context_prompt}"

        self.debug(
            {
                "providers_loaded": {
                    "tools": await self.__tools_provider.describe_loaded(),
                    "mcp_tools": await self.__mcp_provider.describe_loaded(),
                    "skills": self.__skills_provider.describe_loaded(),
                    "agent_delegation": (
                        await self.__agent_delegation_provider.describe_loaded()
                    ),
                    "memory": self.__memory_provider.describe_loaded(),
                    "context": self.__context_provider.describe_loaded(),
                    "instructions": self.__instructions_provider.describe_loaded(),
                    "dispatch": self.__dispatch_provider.describe_loaded(),
                    "message_processing": (
                        self.__message_processing_provider.describe_loaded()
                    ),
                }
            }
        )

        completions = await self.__context_provider.process_conversation(
            initial_context=self.__conversation_context,
            conversation=conversation,
        )

        response = await self.__client.complete(
            bot_setup_description=self.__system_prompt,
            completions=completions,
            tools=self.tools_registry,
            stream=True,
            execute_tools=True,
            stream_tool_events=True,
            concurrent_tool_execution=True,
            log_fn=self.debug,
        )

        async def raw_stream() -> (
            AsyncGenerator[Tuple[ChatResponse, ChatMessage], None]
        ):
            async for chat_client_response in response:
                if chat_client_response.error is not None:
                    raise chat_client_response.error

                message = chat_client_response.to_chat_message()
                if not message:
                    raise Exception("No response from chat completion client")

                yield chat_client_response, message

        async for message in self.__message_processing_provider.process_stream(
            raw_stream()
        ):
            yield message
