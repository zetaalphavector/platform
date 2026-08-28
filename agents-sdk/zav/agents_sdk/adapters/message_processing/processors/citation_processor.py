import re
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.citation import (
    CitationConfiguration,
    CitationContextContribution,
    CitationStore,
)
from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
    ChatCompletionSender,
)
from zav.agents_sdk.adapters.message_processing.message_processor import (
    MessageProcessor,
    StreamItem,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import (
    ChatMessage,
    ChatMessageEvidence,
    ContentPart,
    ConversationContext,
    CustomContext,
    CustomContextItem,
    DocumentContext,
)


class ResolvedCitation(BaseModel):
    evidence: ChatMessageEvidence
    context: CitationContextContribution


class CitationProcessorConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable citation post-processing.")


class CitationProcessor(MessageProcessor):
    """Parses citations from LLM output and enriches messages with evidence.

    Dispatches to strategy-specific logic based on ``CitationConfig``.
    Reads accumulated search results from ``CitationStore`` to validate
    citations, build ``ChatMessageEvidence``, and produce output
    ``DocumentContext`` (cited document IDs for the next turn).
    """

    source_name = "citation_processor"

    _INLINE_URL_PATTERN = re.compile(r'\[\[(\d+)\]\]\(([^)\s]+)(?:\s+"([^"]*)")?\)')
    _SUP_NUMERIC_PATTERN = re.compile(r"<sup>(\d+)</sup>")

    def __init__(
        self,
        citation_config: CitationConfiguration,
        citation_store: CitationStore,
        enabled: bool,
    ):
        self.enabled = enabled
        self.__config = citation_config
        self.__store = citation_store

    async def process_stream(
        self, stream: AsyncGenerator[StreamItem, None]
    ) -> AsyncGenerator[StreamItem, None]:
        last_eligible: Optional[StreamItem] = None

        async for response, message in stream:
            completion = response.chat_completion
            is_eligible = (
                completion is not None
                and completion.sender != ChatCompletionSender.TOOL
                and not completion.function_call_request
                and message.content
                and self.__store.hits
            )

            if last_eligible is not None:
                yield last_eligible

            if is_eligible:
                last_eligible = (response, message)
            else:
                last_eligible = None
                yield response, message

        if last_eligible is not None:
            response, message = last_eligible
            yield response, self.__resolve_citations(message)

    def __resolve_citations(self, message: ChatMessage) -> ChatMessage:
        if self.__config.strategy == "inline_url":
            return self.__process_inline_url(message)
        elif self.__config.strategy == "short_id":
            return self.__process_short_id(message)
        elif self.__config.strategy == "deferred":
            return self.__process_deferred(message)
        elif self.__config.strategy == "sup_numeric":
            return self.__process_sup_numeric(message)
        return message

    def __process_inline_url(self, message: ChatMessage) -> ChatMessage:
        content = message.content
        evidences: List[ChatMessageEvidence] = []
        context = CitationContextContribution()
        content_replacements: List[Tuple[str, str]] = []

        for match in self._INLINE_URL_PATTERN.finditer(content):
            citation_number = match.group(1)
            source_url = match.group(2)
            full_match = match.group(0)
            text_extract = match.group(3)
            anchor_text = f"[[{citation_number}]]"
            registered = self.__resolve_citation(source_url, anchor_text, text_extract)
            if not registered:
                continue

            content = content.replace(full_match, anchor_text, 1)
            content_replacements.append((full_match, anchor_text))
            evidences.append(registered.evidence)
            self.__merge_context(context, registered.context)

        if not evidences:
            return message

        return self.__build_enriched_message(
            message,
            evidences,
            context,
            content=content,
            content_replacements=content_replacements,
        )

    def __process_short_id(self, message: ChatMessage) -> ChatMessage:
        content = message.content
        short_id_map = self.__store.short_id_to_source_url
        if not short_id_map:
            return message

        evidences: List[ChatMessageEvidence] = []
        context = CitationContextContribution()

        bare_pattern = re.compile(r"\[([a-f0-9]{7,8})\]")
        for match in bare_pattern.finditer(content):
            short_id = match.group(1)
            source_url = short_id_map.get(short_id)
            if not source_url:
                continue

            registered = self.__resolve_citation(
                source_url=source_url,
                anchor_text=match.group(0),
                text_extract=None,
            )
            if not registered:
                continue

            evidences.append(registered.evidence)
            self.__merge_context(context, registered.context)

        if not evidences:
            return message

        return self.__build_enriched_message(message, evidences, context)

    def __process_deferred(self, message: ChatMessage) -> ChatMessage:
        # Deferred strategy: match response text against retrieved results.
        # Full implementation requires fuzzy text matching (token_set_ratio).
        # For now, collect all retrieved document IDs as potential citations.
        logger.debug("Deferred citation processing not yet fully implemented")
        return message

    def __process_sup_numeric(self, message: ChatMessage) -> ChatMessage:
        content = message.content
        evidences: List[ChatMessageEvidence] = []
        context = CitationContextContribution()

        hits_by_index = list(self.__store.hits.items())

        for match in self._SUP_NUMERIC_PATTERN.finditer(content):
            citation_number = int(match.group(1))
            idx = citation_number - 1
            if idx < 0 or idx >= len(hits_by_index):
                continue

            citation_key, hit = hits_by_index[idx]
            anchor_text = f"<sup>{citation_number}</sup>"

            resolved = self.__resolve_citation(
                source_url=citation_key,
                anchor_text=anchor_text,
                text_extract=hit.context_hit.get("description", ""),
            )
            if not resolved:
                continue

            evidences.append(resolved.evidence)
            self.__merge_context(context, resolved.context)

        if not evidences:
            return message

        return self.__build_enriched_message(message, evidences, context)

    def __resolve_citation(
        self, source_url: str, anchor_text: str, text_extract: Optional[str]
    ) -> Optional[ResolvedCitation]:
        hit = self.__store.get_hit(source_url)
        if not hit:
            # The model may cite a document id instead of the registered key
            # (e.g. when the id reached it through injected context).
            hit = self.__store.get_hit_by_document_id(source_url)
        if not hit:
            logger.warning(f"Citation could not be resolved: {source_url[:120]}")
            return None

        evidence_url = (
            hit.evidence_url or hit.search_hit.get("document_hit_url") or source_url
        )
        if not evidence_url:
            return None

        return ResolvedCitation(
            evidence=ChatMessageEvidence(
                document_hit_url=evidence_url,
                text_extract=text_extract,
                anchor_text=anchor_text,
            ),
            context=hit.context_contribution,
        )

    @staticmethod
    def __merge_context(
        target: CitationContextContribution,
        source: CitationContextContribution,
    ) -> None:
        target.document_ids.update(source.document_ids)
        target.custom_items.extend(source.custom_items)

    def __build_enriched_message(
        self,
        message: ChatMessage,
        evidences: List[ChatMessageEvidence],
        context: CitationContextContribution,
        content: str = "",
        content_replacements: Optional[List[Tuple[str, str]]] = None,
    ) -> ChatMessage:
        content_parts = self.__replace_content_part_texts(
            list(message.content_parts or []), content_replacements or []
        )
        cited_document_ids = context.document_ids

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

        custom_items = self.__dedupe_custom_items(context.custom_items)
        if custom_items:
            content_parts.append(
                ContentPart(
                    type="context",
                    context=ConversationContext(
                        custom_context=CustomContext(items=custom_items)
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
            reasoning_summary=message.reasoning_summary,
        )

    @staticmethod
    def __replace_content_part_texts(
        content_parts: List[ContentPart],
        replacements: List[Tuple[str, str]],
    ) -> List[ContentPart]:
        if not replacements:
            return content_parts

        result: List[ContentPart] = []
        for part in content_parts:
            if part.type != "text" or not part.text:
                result.append(part)
                continue

            text = part.text
            for original, replacement in replacements:
                text = text.replace(original, replacement)

            result.append(
                ContentPart(
                    type=part.type,
                    context=part.context,
                    tool=part.tool,
                    text=text,
                    table=part.table,
                )
            )
        return result

    @staticmethod
    def __dedupe_custom_items(
        items: List[CustomContextItem],
    ) -> List[CustomContextItem]:
        result: Dict[str, CustomContextItem] = {}
        for item in items:
            result[item.document_id] = item
        return list(result.values())


class CitationProcessorFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        citation_store: CitationStore,
        citation_configuration: CitationConfiguration = CitationConfiguration(),
        citation_processor_configuration: CitationProcessorConfiguration = (
            CitationProcessorConfiguration()
        ),
    ) -> CitationProcessor:
        return CitationProcessor(
            citation_config=citation_configuration,
            citation_store=citation_store,
            enabled=citation_processor_configuration.enabled,
        )
