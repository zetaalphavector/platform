import json
from typing import List, Literal, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ResolvedContextItem,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ConversationContext, FilterContext


class FilterContextSourceConfiguration(BaseModel):
    enabled: bool = Field(True, description="Enable filter context resolution.")


class FilterContextSource(ContextSource):

    source_name = "filters"

    def __init__(self, filter_context: Optional[FilterContext], enabled: bool):
        self.enabled = enabled
        self.__filter_context = filter_context

    def describe(self, origin: ContextOrigin) -> str:
        return (
            "The user has search filters active in the UI. "
            "When searching, include these filters in your search calls."
        )

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        if not self.__filter_context:
            return None
        return [
            ResolvedContextItem(
                id="active_filters",
                data=self.__filter_context.filters,
            )
        ]

    def format_items(
        self,
        items: List[ResolvedContextItem],
        content_format: Literal["json", "xml"],
        max_item_content_length: Optional[int] = None,
    ) -> str:
        if content_format != "xml":
            return json.dumps(
                [item.data for item in items], default=str, ensure_ascii=False
            )
        parts: List[str] = []
        for item in items:
            body = json.dumps(item.data, indent=2, default=str, ensure_ascii=False)
            type_attr = f' type="{item.item_type}"' if item.item_type else ""
            parts.append(f'<item id="{item.id}"{type_attr}>')
            parts.append(body)
            parts.append("</item>")
        return "\n".join(parts)


class FilterContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        conversation_context: ConversationContext,
        filter_context_source_configuration: FilterContextSourceConfiguration = (
            FilterContextSourceConfiguration()
        ),
    ) -> FilterContextSource:
        return FilterContextSource(
            filter_context=(
                conversation_context.filter_context if conversation_context else None
            ),
            enabled=filter_context_source_configuration.enabled,
        )
