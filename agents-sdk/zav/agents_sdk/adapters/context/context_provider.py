import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ContextSourceGroup,
    ResolvedContextItem,
)
from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
    ChatCompletion,
    ChatCompletionSender,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext
from zav.agents_sdk.domain.tools import Tool


class ContextConfiguration(BaseModel):
    """Configuration for the context provider."""

    enabled: bool = Field(True, description="Enable conversation context resolution.")
    injection_mode: Literal["prompt", "conversation", "both"] = Field(
        "conversation",
        description=(
            "Where to inject resolved context: "
            "'prompt' (system prompt only), "
            "'conversation' (as developer messages interleaved with the conversation), "
            "or 'both'."
        ),
    )
    content_format: Literal["json", "xml"] = Field(
        "xml",
        description=(
            "Serialization format for context: "
            "'json' (raw JSON dump) or "
            "'xml' (structured XML tags per source with per-item ids, similar to how "
            "GitHub Copilot formats attachments)."
        ),
    )
    max_item_content_length: Optional[int] = Field(
        None,
        description=(
            "Maximum character length for a single item's serialized content. "
            "When exceeded, the item is marked as summarized and its body is "
            "omitted. Set to None to disable truncation."
        ),
    )


@dataclass
class _ResolvedSourceBlock:
    source: ContextSource
    description: str
    items: List[ResolvedContextItem]


class ContextProvider:
    """Resolves conversation context through registered sources.

    Delegates to ``ContextSource`` instances to interpret what the user is
    communicating indirectly (attached documents, active tags, custom data)
    and surfaces the resolved data to the agent via the system prompt
    and/or developer messages in the conversation.
    """

    def __init__(
        self,
        sources: List[ContextSource],
        enabled: bool,
        injection_mode: Literal["prompt", "conversation", "both"],
        content_format: Literal["json", "xml"],
        max_item_content_length: Optional[int],
    ):
        self.__sources = sources
        self.__enabled = enabled
        self.__injection_mode = injection_mode
        self.__content_format = content_format
        self.__max_item_content_length = max_item_content_length

    async def get_tools(self) -> List[Tool]:
        if not self.__enabled:
            return []
        return []

    async def to_prompt(
        self, initial_context: Optional[ConversationContext] = None
    ) -> str:
        if not self.__enabled:
            return ""

        if self.__injection_mode not in ("prompt", "both"):
            return ""

        if not initial_context or initial_context.is_empty():
            return ""

        blocks = await self.__resolve_sources(initial_context, ContextOrigin.INITIAL)
        if not blocks:
            return ""

        return self.__format(blocks)

    async def process_conversation(
        self,
        initial_context: Optional[ConversationContext],
        conversation: List[ChatMessage],
    ) -> List[ChatCompletion]:
        if not self.__enabled:
            return [ChatCompletion.from_chat_message(m) for m in conversation]

        completions: List[ChatCompletion] = []

        if self.__injection_mode in ("conversation", "both"):
            if initial_context and not initial_context.is_empty():
                completion = await self.__to_completion(
                    initial_context, ContextOrigin.INITIAL
                )
                if completion:
                    completions.append(completion)

        for message in conversation:
            text_completion = ChatCompletion.from_chat_message(message)

            if message.content_parts:
                for content_part in message.content_parts:
                    ctx = content_part.context
                    if ctx and not ctx.is_empty():
                        if self.__injection_mode in ("conversation", "both"):
                            completion = await self.__to_completion(
                                ctx, ContextOrigin.CONTENT_PART
                            )
                            if completion:
                                completions.append(completion)

            completions.append(text_completion)

        return completions

    async def __resolve_sources(
        self, context: ConversationContext, origin: ContextOrigin
    ) -> List[_ResolvedSourceBlock]:
        blocks: List[_ResolvedSourceBlock] = []
        for source in self.__sources:
            try:
                items = await source.resolve(context)
                if items:
                    blocks.append(
                        _ResolvedSourceBlock(
                            source=source,
                            description=source.describe(origin),
                            items=items,
                        )
                    )
            except Exception as e:
                logger.error(f"Context source '{source.source_name}' failed: {e}")
        return blocks

    def __format(self, blocks: List[_ResolvedSourceBlock]) -> str:
        if self.__content_format == "xml":
            return self.__format_xml(blocks)
        return self.__format_json(blocks)

    def __format_json(self, blocks: List[_ResolvedSourceBlock]) -> str:
        merged: Dict[str, Any] = {}
        for block in blocks:
            merged[block.source.source_name] = [item.data for item in block.items]
        return json.dumps(merged, default=str, ensure_ascii=False)

    def __format_xml(self, blocks: List[_ResolvedSourceBlock]) -> str:
        parts: List[str] = ["<context>"]
        for block in blocks:
            parts.append(f'<source name="{block.source.source_name}">')
            parts.append(block.description)
            formatted = block.source.format_items(
                block.items,
                content_format="xml",
                max_item_content_length=self.__max_item_content_length,
            )
            parts.append(formatted)
            parts.append("</source>")
        parts.append("</context>")
        return "\n".join(parts)

    async def __to_completion(
        self, context: ConversationContext, origin: ContextOrigin
    ) -> Optional[ChatCompletion]:
        blocks = await self.__resolve_sources(context, origin)
        if not blocks:
            return None
        return ChatCompletion(
            sender=ChatCompletionSender.DEVELOPER,
            content=self.__format(blocks),
        )


class ContextProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        context_source_group: ContextSourceGroup = ContextSourceGroup(items=[]),
        context_configuration: ContextConfiguration = ContextConfiguration(),
    ) -> ContextProvider:
        return ContextProvider(
            sources=context_source_group.items,
            enabled=context_configuration.enabled,
            injection_mode=context_configuration.injection_mode,
            content_format=context_configuration.content_format,
            max_item_content_length=context_configuration.max_item_content_length,
        )
