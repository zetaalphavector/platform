import hashlib
from typing import Dict, Optional

from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class ConversationImageStore:
    """Accumulates image data from conversation messages during a request.

    Context sources call ``register_image()`` for each image found in
    the conversation.  Tools later call ``get_image_data()`` to recover
    the original base64 data from a short text ID the LLM can reference.

    Follows the same singleton-per-request pattern as ``CitationStore``.
    """

    def __init__(self) -> None:
        self.__images: Dict[str, str] = {}

    def register_image(self, image_data: str) -> str:
        short_id = hashlib.sha256(image_data.encode()).hexdigest()[:8]
        self.__images[short_id] = image_data
        return short_id

    def get_image_data(self, short_id: str) -> Optional[str]:
        return self.__images.get(short_id)

    @property
    def images(self) -> Dict[str, str]:
        return dict(self.__images)


class ConversationImageStoreFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(cls) -> ConversationImageStore:
        return ConversationImageStore()
