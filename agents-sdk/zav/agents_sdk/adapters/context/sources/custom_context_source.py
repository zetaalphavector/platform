from typing import List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ResolvedContextItem,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ConversationContext


class CustomContextSourceConfiguration(BaseModel):
    enabled: bool = Field(True, description="Enable custom context pass-through.")


class CustomContextSource(ContextSource):
    """Pass-through for custom context items."""

    source_name = "custom"

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def describe(self, origin: ContextOrigin) -> str:
        if origin == ContextOrigin.INITIAL:
            return "Custom context for this conversation:"
        return "Custom context added by the user in this message:"

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        custom_ctx = context.custom_context
        if not custom_ctx or not custom_ctx.items:
            return None

        return [
            ResolvedContextItem(
                id=str(i), data={"content": item.content}, item_type="custom"
            )
            for i, item in enumerate(custom_ctx.items)
        ]


class CustomContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        custom_context_source_configuration: CustomContextSourceConfiguration = (
            CustomContextSourceConfiguration()
        ),
    ) -> CustomContextSource:
        return CustomContextSource(
            enabled=custom_context_source_configuration.enabled,
        )
