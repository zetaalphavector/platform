import base64
from typing import Any, Dict, List, Optional

import pymupdf
from jinja2 import Template
from zav.logging import logger
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.agent_state.conversation_image_store import (
    ConversationImageStore,
)
from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
    ZAVChatCompletionClient,
)
from zav.agents_sdk.adapters.retrievers.zav_retriever import ZAVRetriever
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage, ChatMessageSender
from zav.agents_sdk.domain.tools import Tool, hide, streamable

_DEFAULT_VALIDATION_PROMPT = """\
You are comparing two images to determine if they are related.

The first image is the user's input image.
The second image is a page from a retrieved document.
{% if document_title %}
Document title: {{ document_title }}
{% endif %}
Carefully compare the visual content of both images, including any \
shapes, diagrams, text, labels, numbers, and other distinguishing features.

Provide a brief analysis covering:
1. What you see in each image
2. Specific visual elements that match or differ
3. Your conclusion on whether the document page is relevant to \
the user's image and why\
"""


def _render_pdf_page_to_base64(
    pdf_bytes: bytes, page_number: int, dpi: int = 96
) -> Optional[str]:
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_idx = page_number - 1
            if page_idx < 0 or page_idx >= doc.page_count:
                return None
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("ascii")
            return f"data:image/png;base64,{b64}"
    except Exception:
        logger.warning("Failed to render PDF page %d", page_number, exc_info=True)
        return None


class ImageToolsSourceConfiguration(BaseModel):
    enabled: bool = False
    validation_prompt: str = _DEFAULT_VALIDATION_PROMPT
    render_dpi: int = 96


class ImageToolsSource(ToolsSource):
    """Provides image validation tools for comparing user images against
    document pages.

    Uses ``ConversationImageStore`` (shared singleton with
    ``ImageContextSource``) to resolve image IDs, ``ZAVRetriever``
    to fetch document PDFs, and ``ZAVChatCompletionClient`` to perform
    visual comparison via a multimodal LLM call.
    """

    source_name = "image_tools"

    def __init__(
        self,
        conversation_image_store: ConversationImageStore,
        zav_retriever: ZAVRetriever,
        zav_chat_completion_client: ZAVChatCompletionClient,
        image_tools_source_configuration: ImageToolsSourceConfiguration,
    ):
        self.__image_store = conversation_image_store
        self.__retriever = zav_retriever
        self.__chat_client = zav_chat_completion_client
        self.__validation_template = Template(
            image_tools_source_configuration.validation_prompt
        )
        self.enabled = image_tools_source_configuration.enabled
        self.__render_dpi = image_tools_source_configuration.render_dpi

    async def get_tools(self) -> List[Tool]:
        return [
            Tool.from_callable(
                name="validate_image_hit",
                executable=self.validate_image_hit,
            ),
        ]

    @streamable(
        running_text="Validating image against document...",
        completed_text="Validated image against document.",
        params_transform=hide,
        response_transform=hide,
    )
    async def validate_image_hit(
        self,
        image_id: str,
        document_id: str,
        page_number: int = 1,
        document_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Visually compare a user-uploaded image against a page of a retrieved
        document.

        Fetches the document PDF, renders the specified page, and uses a
        multimodal LLM to analyze how the page content relates to the
        user's image.

        Args:
            image_id: The short ID of the user's image from the conversation.
            document_id: The document ID from search results (the base ID
                without chunk suffix).
            page_number: The 1-based page number to render and compare.
            document_title: Optional document title for additional context.

        Returns:
            Dict with 'analysis' (the LLM's comparison), 'document_id',
            and 'page_number'.
        """
        input_image = self.__image_store.get_image_data(image_id)
        if input_image is None:
            return {"error": f"Image ID {image_id} not found in conversation"}

        try:
            pdf_bytes = await self.__retriever.get_pdf_asset(document_id)
        except Exception as e:
            logger.debug("get_pdf_asset raised for %s: %s", document_id, e)
            pdf_bytes = None

        if not pdf_bytes:
            return {
                "error": f"Could not fetch PDF for document {document_id}",
                "document_id": document_id,
                "page_number": page_number,
            }

        page_image = _render_pdf_page_to_base64(
            pdf_bytes, page_number, self.__render_dpi
        )
        if page_image is None:
            return {
                "error": f"Could not render page {page_number} "
                f"of document {document_id}",
                "document_id": document_id,
                "page_number": page_number,
            }

        prompt = self.__validation_template.render(
            document_title=document_title,
        )

        messages = [
            ChatMessage(
                sender=ChatMessageSender.USER,
                content="Input image (the image the user is looking for):",
                image_uri=input_image,
            ),
            ChatMessage(
                sender=ChatMessageSender.USER,
                content="Retrieved document page:",
                image_uri=page_image,
            ),
            ChatMessage(
                sender=ChatMessageSender.USER,
                content=prompt,
            ),
        ]

        try:
            chat_response = await self.__chat_client.complete(messages=messages)
        except Exception as e:
            logger.warning("Visual validation LLM call failed: %s", e, exc_info=True)
            return {
                "analysis": f"Validation unavailable: {e}",
                "document_id": document_id,
                "page_number": page_number,
            }

        if chat_response.error is not None or chat_response.chat_completion is None:
            error_detail = chat_response.error or "no response from comparison model"
            return {
                "analysis": f"Validation unavailable: {error_detail}",
                "document_id": document_id,
                "page_number": page_number,
            }

        return {
            "analysis": chat_response.chat_completion.content.strip(),
            "document_id": document_id,
            "page_number": page_number,
        }


class ImageToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        conversation_image_store: ConversationImageStore,
        zav_retriever: ZAVRetriever,
        zav_chat_completion_client: ZAVChatCompletionClient,
        image_tools_source_configuration: ImageToolsSourceConfiguration = (
            ImageToolsSourceConfiguration()
        ),
    ) -> ImageToolsSource:
        return ImageToolsSource(
            conversation_image_store=conversation_image_store,
            zav_retriever=zav_retriever,
            zav_chat_completion_client=zav_chat_completion_client,
            image_tools_source_configuration=image_tools_source_configuration,
        )
