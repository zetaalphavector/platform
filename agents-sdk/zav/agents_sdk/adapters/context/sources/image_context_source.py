from typing import List, Optional

from jinja2 import Template
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.conversation_image_store import (
    ConversationImageStore,
)
from zav.agents_sdk.adapters.context.context_source import (
    ContextOrigin,
    ContextSource,
    ResolvedContextItem,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext

_DEFAULT_IMAGE_ID_TEXT = "[Image ID: {{ image_id }}]"

_IMAGE_CONTEXT_DESCRIPTION = (
    "The user has shared images in this conversation. "
    "Each image has been assigned a short ID you can reference "
    "when calling image-related tools (e.g. search_from_image)."
)


class ImageContextSourceConfiguration(BaseModel):
    enabled: bool = Field(True, description="Enable image context resolution.")


class ImageContextSource(ContextSource):

    source_name = "conversation_images"

    def __init__(
        self,
        conversation_image_store: ConversationImageStore,
        enabled: bool,
        image_id_text: str = _DEFAULT_IMAGE_ID_TEXT,
    ):
        self.enabled = enabled
        self.__store = conversation_image_store
        self.__template = Template(image_id_text)

    def describe(self, origin: ContextOrigin) -> str:
        return _IMAGE_CONTEXT_DESCRIPTION

    async def resolve(
        self, context: ConversationContext
    ) -> Optional[List[ResolvedContextItem]]:
        return None

    async def resolve_from_conversation(
        self,
        conversation: List[ChatMessage],
    ) -> Optional[List[ResolvedContextItem]]:
        items: List[ResolvedContextItem] = []

        for message in conversation:
            if not message.image_uri:
                continue

            short_id = self.__store.register_image(message.image_uri)
            display_text = self.__template.render(image_id=short_id)
            items.append(
                ResolvedContextItem(
                    id=short_id,
                    data={
                        "image_id": short_id,
                        "display_text": display_text,
                    },
                    item_type="image",
                )
            )

        return items if items else None


class ImageContextSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        conversation_image_store: ConversationImageStore,
        image_id_text: str = _DEFAULT_IMAGE_ID_TEXT,
        image_context_source_configuration: ImageContextSourceConfiguration = ImageContextSourceConfiguration(),  # noqa: E501
    ) -> ImageContextSource:
        return ImageContextSource(
            conversation_image_store=conversation_image_store,
            image_id_text=image_id_text,
            enabled=image_context_source_configuration.enabled,
        )
