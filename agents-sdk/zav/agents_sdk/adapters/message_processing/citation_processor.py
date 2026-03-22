import re
from typing import List, Set

from zav.logging import logger

from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
    ChatCompletionSender,
    ChatResponse,
)
from zav.agents_sdk.adapters.message_processing.message_processor import (
    MessageProcessor,
)
from zav.agents_sdk.adapters.search.citation import CitationConfig, CitationStore
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import (
    ChatMessage,
    ChatMessageEvidence,
    ContentPart,
    ConversationContext,
    DocumentContext,
)


class CitationProcessor(MessageProcessor):
    """Parses citations from LLM output and enriches messages with evidence.

    Dispatches to strategy-specific logic based on ``CitationConfig``.
    Reads accumulated search results from ``CitationStore`` to validate
    citations, build ``ChatMessageEvidence``, and produce output
    ``DocumentContext`` (cited document IDs for the next turn).
    """

    source_name = "citation_processor"

    _INLINE_URL_PATTERN = re.compile(r'\[\[(\d+)\]\]\(([^)\s]+)(?:\s+"([^"]*)")?\)')

    def __init__(
        self,
        citation_config: CitationConfig,
        citation_store: CitationStore,
    ):
        self.__config = citation_config
        self.__store = citation_store

    async def process(
        self, response: ChatResponse, message: ChatMessage
    ) -> ChatMessage:
        completion = response.chat_completion
        if completion is None:
            return message
        if completion.sender == ChatCompletionSender.TOOL:
            return message
        if completion.function_call_request:
            return message

        if not message.content:
            return message
        if not self.__store.hits:
            return message

        if self.__config.strategy == "inline_url":
            return self.__process_inline_url(message)
        elif self.__config.strategy == "short_id":
            return self.__process_short_id(message)
        elif self.__config.strategy == "deferred":
            return self.__process_deferred(message)
        return message

    def __process_inline_url(self, message: ChatMessage) -> ChatMessage:
        content = message.content
        evidences: List[ChatMessageEvidence] = []
        cited_document_ids: Set[str] = set()

        for match in self._INLINE_URL_PATTERN.finditer(content):
            citation_number = match.group(1)
            source_url = match.group(2)
            full_match = match.group(0)
            text_extract = match.group(3)
            anchor_text = f"[[{citation_number}]]"

            registered = self.__store.get_hit(source_url)
            if not registered:
                continue

            content = content.replace(full_match, anchor_text, 1)

            evidences.append(
                ChatMessageEvidence(
                    document_hit_url=registered.search_hit.get(
                        "document_hit_url", source_url
                    ),
                    text_extract=text_extract,
                    anchor_text=anchor_text,
                )
            )

            doc_id = registered.search_hit.get("id", "")
            if doc_id:
                cited_document_ids.add(doc_id)

        if not evidences:
            return message

        return self.__build_enriched_message(
            message, evidences, cited_document_ids, content=content
        )

    def __process_short_id(self, message: ChatMessage) -> ChatMessage:
        content = message.content
        short_id_map = self.__store.short_id_to_source_url
        if not short_id_map:
            return message

        evidences: List[ChatMessageEvidence] = []
        cited_document_ids: Set[str] = set()

        bare_pattern = re.compile(r"\[([a-f0-9]{7,8})\]")
        for match in bare_pattern.finditer(content):
            short_id = match.group(1)
            source_url = short_id_map.get(short_id)
            if not source_url:
                continue

            registered = self.__store.get_hit(source_url)
            if not registered:
                continue

            evidences.append(
                ChatMessageEvidence(
                    document_hit_url=registered.search_hit.get(
                        "document_hit_url", source_url
                    ),
                    anchor_text=match.group(0),
                )
            )

            doc_id = registered.search_hit.get("id", "")
            if doc_id:
                cited_document_ids.add(doc_id)

        if not evidences:
            return message

        return self.__build_enriched_message(message, evidences, cited_document_ids)

    def __process_deferred(self, message: ChatMessage) -> ChatMessage:
        # Deferred strategy: match response text against retrieved results.
        # Full implementation requires fuzzy text matching (token_set_ratio).
        # For now, collect all retrieved document IDs as potential citations.
        logger.debug("Deferred citation processing not yet fully implemented")
        return message

    def __build_enriched_message(
        self,
        message: ChatMessage,
        evidences: List[ChatMessageEvidence],
        cited_document_ids: Set[str],
        content: str = "",
    ) -> ChatMessage:
        content_parts = list(message.content_parts or [])
        if cited_document_ids:
            content_parts.append(
                ContentPart(
                    type="context",
                    context=ConversationContext(
                        document_context=DocumentContext(
                            document_ids=list(cited_document_ids),
                            retrieval_unit="chunk",
                        )
                    ),
                )
            )

        return ChatMessage(
            sender=message.sender,
            content=content or message.content,
            message_id=message.message_id,
            content_parts=content_parts or None,
            image_uri=message.image_uri,
            function_call_request=message.function_call_request,
            function_call_response=message.function_call_response,
            evidences=evidences,
            function_specs=message.function_specs,
        )


class CitationProcessorFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        citation_store: CitationStore,
        citation_configuration: CitationConfig = CitationConfig(),
    ) -> CitationProcessor:
        return CitationProcessor(
            citation_config=citation_configuration,
            citation_store=citation_store,
        )
