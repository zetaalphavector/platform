import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, List, Literal, Optional

from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.chat_message import ConversationContext


class ContextOrigin(str, Enum):
    INITIAL = "initial"
    CONTENT_PART = "content_part"


@dataclass
class ResolvedContextItem:
    id: str
    data: Dict[str, Any]
    item_type: str = ""


class ContextSource(ABC):
    """Abstract source for resolving one type of conversation context.

    Extend this to teach agents about platform-specific context such as
    attached documents, tags, or custom data the user is interacting with.
    """

    source_name: ClassVar[str]

    @abstractmethod
    def describe(self, origin: ContextOrigin) -> str:
        """Return a human-readable description for this source given the
        origin. The description explains to the agent what this context
        means — e.g. the user is viewing a document page versus attaching
        a document in a message."""
        raise NotImplementedError

    @abstractmethod
    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        """Resolve this source's portion of context into a list of items.

        Each item has a normalised ``id`` (how the agent should refer to
        this object) and ``data`` (the source-specific payload).

        Returns ``None`` if this source has nothing to contribute.
        """
        raise NotImplementedError

    def format_items(
        self,
        items: List[ResolvedContextItem],
        content_format: Literal["json", "xml"],
        max_item_content_length: Optional[int] = None,
    ) -> str:
        """Serialize resolved items into the target format.

        Override this in subclasses to control source-specific formatting
        (e.g. nested XML for tags with sub-documents).

        The default implementation serializes each item as a flat JSON body
        wrapped in ``<item>`` tags (xml) or as a JSON array (json).
        """
        if content_format == "xml":
            return self._default_format_xml(items, max_item_content_length)
        return json.dumps(
            [item.data for item in items], default=str, ensure_ascii=False
        )

    @staticmethod
    def _default_format_xml(
        items: List[ResolvedContextItem],
        max_item_content_length: Optional[int] = None,
    ) -> str:
        parts: List[str] = []
        for item in items:
            body = json.dumps(item.data, indent=2, default=str, ensure_ascii=False)
            is_summarized = (
                max_item_content_length is not None
                and len(body) > max_item_content_length
            )
            type_attr = f' type="{item.item_type}"' if item.item_type else ""
            if is_summarized:
                parts.append(f'<item id="{item.id}"{type_attr} isSummarized="true"/>')
            else:
                parts.append(f'<item id="{item.id}"{type_attr}>')
                parts.append(body)
                parts.append("</item>")
        return "\n".join(parts)


class ContextSourceGroup(DependencyGroup[ContextSource]):
    """Collects all registered ``ContextSource`` subclass instances."""

    __collects__ = ContextSource
