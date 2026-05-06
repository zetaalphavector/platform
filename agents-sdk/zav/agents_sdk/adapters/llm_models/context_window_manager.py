import json
import uuid
from typing import Any, Dict, List, Optional

from zav.logging import logger
from zav.prompt_completion import ChatMessage as PcChatMessage
from zav.prompt_completion import ChatMessageSender as PcChatMessageSender
from zav.prompt_completion import ToolCallResponse as PcToolCallResponse
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.token_estimation import count_tokens, truncate_to_tokens
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool

_DEFAULT_RETRIEVAL_LIMIT = 40_000

_RETRIEVAL_HINT = (
    "\n\n---\n\n[Rest of the output is virtualized. "
    'When needed, use retrieve_tool_output(tool_call_id="{tool_call_id}", '
    "offset=0, limit={limit}) "
    "to read sections."
    "\n\nFull output: {total_chars:,} chars (~{total_tokens:,} tokens). "
    "Context budget: {budget:,} tokens, ~{remaining:,} remaining.]"
)

_BOT_RETRIEVAL_HINT = (
    "\n\n---\n\n[Rest of the output is virtualized. "
    'When needed, use retrieve_message(message_id="{message_id}", '
    "offset=0, limit={limit}) "
    "to read sections."
    "\n\nFull response: {total_chars:,} chars (~{total_tokens:,} tokens). "
    "Context budget: {budget:,} tokens, ~{remaining:,} remaining.]"
)


class ContextWindowConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description="Enable automatic context window management.",
    )
    max_context_tokens: int = Field(
        120_000,
        description=(
            "Target token budget. When the conversation exceeds this, "
            "old tool outputs and bot responses are replaced with "
            "truncated previews and retrieval pointers."
        ),
    )
    preview_tokens: int = Field(
        200,
        description=(
            "Number of tokens to keep as a preview when replacing "
            "a tool output or bot response with a retrieval pointer."
        ),
    )


def _estimate_tokens(message: PcChatMessage) -> int:
    total = count_tokens(message.content or "")
    if message.content_parts:
        for part in message.content_parts:
            total += count_tokens(part.text or "")
            if part.image:
                total += count_tokens(part.image.image_uri or "")
            if part.document:
                total += count_tokens(part.document.document_base64 or "")
    if message.image_uri:
        total += count_tokens(message.image_uri)
    if message.tool_call_requests:
        for tcr in message.tool_call_requests:
            total += count_tokens(tcr.function_call_request.function_name or "")
            params = tcr.function_call_request.function_params
            if params:
                total += count_tokens(json.dumps(params, default=str))
    if message.tool_call_responses:
        for tcr in message.tool_call_responses:
            total += count_tokens(tcr.tool_response or "")
    if message.function_call_request:
        total += count_tokens(message.function_call_request.function_name or "")
        fn_params = message.function_call_request.function_params
        if fn_params:
            total += count_tokens(json.dumps(fn_params, default=str))
    if message.function_call_response:
        total += count_tokens(message.function_call_response.function_response or "")
    return total


def _make_preview(text: str, max_tokens: int, hint: str) -> str:
    preview = truncate_to_tokens(text, max_tokens)
    return preview + hint


def _replace_tool_responses(
    message: PcChatMessage,
    store: Dict[str, str],
    preview_tokens: int,
    budget: int,
    remaining: int,
) -> PcChatMessage:
    if not message.tool_call_responses:
        return message

    new_responses: list[PcToolCallResponse] = []
    for tcr in message.tool_call_responses:
        full_text = tcr.tool_response or ""
        tool_call_id = tcr.id
        store[tool_call_id] = full_text
        total_chars = len(full_text)
        total_tokens = count_tokens(full_text)
        suggested_limit = min(_DEFAULT_RETRIEVAL_LIMIT, total_chars)
        hint = _RETRIEVAL_HINT.format(
            tool_call_id=tool_call_id,
            total_chars=total_chars,
            total_tokens=total_tokens,
            budget=budget,
            remaining=remaining,
            limit=suggested_limit,
        )
        preview = _make_preview(full_text, preview_tokens, hint)
        new_responses.append(PcToolCallResponse(id=tool_call_id, tool_response=preview))

    return PcChatMessage(
        sender=message.sender,
        content=message.content,
        image_uri=message.image_uri,
        content_parts=message.content_parts,
        function_call_request=message.function_call_request,
        function_call_response=message.function_call_response,
        tool_call_requests=message.tool_call_requests,
        tool_call_responses=new_responses,
    )


def _replace_bot_content(
    message: PcChatMessage,
    store: Dict[str, str],
    preview_tokens: int,
    budget: int,
    remaining: int,
) -> PcChatMessage:
    full_text = message.content or ""
    message_id = str(uuid.uuid4())
    store[message_id] = full_text
    total_chars = len(full_text)
    total_tokens = count_tokens(full_text)
    suggested_limit = min(_DEFAULT_RETRIEVAL_LIMIT, total_chars)
    hint = _BOT_RETRIEVAL_HINT.format(
        message_id=message_id,
        total_chars=total_chars,
        total_tokens=total_tokens,
        budget=budget,
        remaining=remaining,
        limit=suggested_limit,
    )
    preview = _make_preview(full_text, preview_tokens, hint)

    return PcChatMessage(
        sender=message.sender,
        content=preview,
        image_uri=message.image_uri,
        content_parts=message.content_parts,
        function_call_request=message.function_call_request,
        function_call_response=message.function_call_response,
        tool_call_requests=message.tool_call_requests,
        tool_call_responses=message.tool_call_responses,
    )


def _collect_compactable_entries(
    messages: List[PcChatMessage],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    for msg_idx, message in enumerate(messages):
        if message.tool_call_responses:
            for resp_idx, tcr in enumerate(message.tool_call_responses):
                text = tcr.tool_response or ""
                tokens = count_tokens(text)
                if tokens > 0:
                    entries.append(
                        {
                            "msg_idx": msg_idx,
                            "resp_idx": resp_idx,
                            "type": "tool_response",
                            "tokens": tokens,
                            "text": text,
                            "id": tcr.id,
                        }
                    )

        if (
            message.sender == PcChatMessageSender.BOT
            and message.content
            and not message.tool_call_requests
        ):
            tokens = count_tokens(message.content)
            if tokens > 0:
                entries.append(
                    {
                        "msg_idx": msg_idx,
                        "type": "bot_content",
                        "tokens": tokens,
                        "text": message.content,
                    }
                )

    return entries


class ContextWindowManager:
    def __init__(self, config: ContextWindowConfiguration):
        self.__config = config
        self.__store: Dict[str, str] = {}
        self.__last_estimated_tokens: int = 0

    @property
    def store(self) -> Dict[str, str]:
        return self.__store

    def needs_compaction(
        self,
        messages: List[PcChatMessage],
        bot_setup_description: Optional[str] = None,
    ) -> bool:
        if not self.__config.enabled:
            return False
        total = sum(_estimate_tokens(m) for m in messages)
        if bot_setup_description:
            total += count_tokens(bot_setup_description)
        self.__last_estimated_tokens = total
        return total > self.__config.max_context_tokens

    async def compact(
        self,
        messages: List[PcChatMessage],
        system_prompt: Optional[str] = None,
    ) -> List[PcChatMessage]:
        total_before = self.__last_estimated_tokens
        budget = self.__config.max_context_tokens
        preview_tokens = self.__config.preview_tokens

        logger.info(
            "Context window management: %d messages, ~%d tokens (budget: %d tokens)",
            len(messages),
            total_before,
            budget,
        )

        entries = _collect_compactable_entries(messages)
        entries.sort(key=lambda e: e["msg_idx"])

        tokens_to_free = total_before - budget
        if tokens_to_free <= 0:
            return messages

        result = list(messages)
        freed = 0
        compacted_msg_indices: set[int] = set()

        for entry in entries:
            if freed >= tokens_to_free:
                break

            msg_idx = entry["msg_idx"]
            entry_tokens = entry["tokens"]
            preview_cost = preview_tokens + 20

            if entry_tokens <= preview_cost:
                continue

            savings = entry_tokens - preview_cost
            remaining_after = max(0, budget - (total_before - freed - savings))

            if entry["type"] == "tool_response":
                result[msg_idx] = _replace_tool_responses(
                    result[msg_idx],
                    self.__store,
                    preview_tokens,
                    budget,
                    remaining_after,
                )
            else:
                result[msg_idx] = _replace_bot_content(
                    result[msg_idx],
                    self.__store,
                    preview_tokens,
                    budget,
                    remaining_after,
                )

            freed += savings
            compacted_msg_indices.add(msg_idx)

        total_after = sum(_estimate_tokens(m) for m in result)
        logger.info(
            "Context window compacted: %d messages unchanged, "
            "%d messages compacted (%d payloads replaced with pointers), "
            "~%d -> ~%d tokens (%.0f%% reduction), "
            "%d outputs stored for retrieval",
            len(result) - len(compacted_msg_indices),
            len(compacted_msg_indices),
            len(self.__store),
            total_before,
            total_after,
            (1 - total_after / total_before) * 100 if total_before > 0 else 0,
            len(self.__store),
        )

        return result

    def retrieve(
        self, identifier: str, offset: int = 0, limit: int = _DEFAULT_RETRIEVAL_LIMIT
    ) -> Optional[str]:
        content = self.__store.get(identifier)
        if content is None:
            return None
        chunk = content[offset : offset + limit]
        total_chars = len(content)
        end = offset + len(chunk)
        if end < total_chars:
            chunk += (
                f"\n\n[Showing chars {offset:,}-{end:,} of {total_chars:,}. "
                f"Call again with offset={end} to continue reading.]"
            )
        return chunk

    def get_tools(self) -> List[Tool]:
        if not self.__config.enabled or not self.__store:
            return []

        def retrieve_tool_output(
            tool_call_id: str,
            offset: int = 0,
            limit: int = _DEFAULT_RETRIEVAL_LIMIT,
        ) -> str:
            """Retrieve a section of a previously truncated tool call output.

            When context window management replaces old tool outputs with
            previews, use this tool to read the original output in chunks.
            Use offset and limit to control which section to read.

            Args:
                tool_call_id: The tool call ID shown in the truncation notice.
                offset: Character offset to start reading from (default: 0).
                limit: Maximum characters to return (default: 40000).
            """
            result = self.retrieve(tool_call_id, offset, limit)
            if result is None:
                return f"No stored output found for tool_call_id='{tool_call_id}'"
            return result

        def retrieve_message(
            message_id: str,
            offset: int = 0,
            limit: int = _DEFAULT_RETRIEVAL_LIMIT,
        ) -> str:
            """Retrieve a section of a previously truncated message.

            When context window management replaces old bot responses with
            previews, use this tool to read the original response in chunks.
            Use offset and limit to control which section to read.

            Args:
                message_id: The message ID shown in the truncation notice.
                offset: Character offset to start reading from (default: 0).
                limit: Maximum characters to return (default: 40000).
            """
            result = self.retrieve(message_id, offset, limit)
            if result is None:
                return f"No stored message found for message_id='{message_id}'"
            return result

        return [
            Tool.from_callable(
                executable=retrieve_tool_output,
                name="retrieve_tool_output",
                description=(
                    "Retrieve a section of a previously truncated tool call output. "
                    "Use offset and limit to read specific sections."
                ),
                # streaming_config=ToolStreamingConfig(
                #     running_text="Retrieving previous tool memory...",
                #     completed_text="Retrieved previous tool memory",
                # ),
            ),
            Tool.from_callable(
                executable=retrieve_message,
                name="retrieve_message",
                description=(
                    "Retrieve a section of a previously truncated message. "
                    "Use offset and limit to read specific sections."
                ),
                # streaming_config=ToolStreamingConfig(
                #     running_text="Retrieving previous message memory...",
                #     completed_text="Retrieved previous message memory",
                # ),
            ),
        ]


class ContextWindowManagerFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        context_window_configuration: ContextWindowConfiguration = (
            ContextWindowConfiguration()
        ),
    ) -> ContextWindowManager:
        return ContextWindowManager(config=context_window_configuration)
