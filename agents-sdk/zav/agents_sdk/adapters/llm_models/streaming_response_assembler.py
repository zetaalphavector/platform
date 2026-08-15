from typing import Dict, List, Optional

from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
    ChatResponse,
)
from zav.agents_sdk.domain.chat_message import ContentPart, ContentPartTool


class StreamingResponseAssembler:
    def __init__(self) -> None:
        """Track text and tool content parts while a response streams."""
        self.__content_parts: List[ContentPart] = []
        self.__current_text_part_index: Optional[int] = None
        self.__tool_part_indexes_by_id: Dict[str, int] = {}

    @property
    def content_parts(self) -> List[ContentPart]:
        """Return the assembled content parts in display order."""
        return list(self.__content_parts)

    @property
    def tool_content_parts(self) -> List[ContentPart]:
        """Return the assembled content parts that represent tool activity."""
        return [part for part in self.__content_parts if part.tool]

    @property
    def has_tool_parts(self) -> bool:
        """Return whether any tool content parts have been recorded."""
        return bool(self.__tool_part_indexes_by_id)

    def record_tool_event(self, event: ContentPartTool) -> None:
        """Record or update the content part for a streamed tool event."""
        self.__record_tool_part(event)

    def reset(self) -> None:
        """Clear all accumulated text and tool content parts."""
        self.__content_parts = []
        self.__current_text_part_index = None
        self.__tool_part_indexes_by_id = {}

    def reset_text(self) -> None:
        """Drop text parts; keep accumulated tool parts."""
        new_parts: List[ContentPart] = []
        new_tool_indexes: Dict[str, int] = {}
        for part in self.__content_parts:
            if part.tool is not None:
                tool_id = part.tool.tool_call_id
                new_tool_indexes[tool_id] = len(new_parts)
                new_parts.append(part)
        self.__content_parts = new_parts
        self.__current_text_part_index = None
        self.__tool_part_indexes_by_id = new_tool_indexes

    def finish_text_segment(self) -> None:
        """Finalize the current text segment before another content part starts."""
        text_part_index = self.__current_text_part_index
        if text_part_index is not None:
            text_part = self.__content_parts[text_part_index]
            if text_part.text:
                self.__content_parts[text_part_index] = ContentPart(
                    type="text", text=f"{text_part.text}\n\n"
                )
        self.__current_text_part_index = None

    def record_response(self, response: ChatResponse) -> None:
        """Record text content from a streamed chat response chunk."""
        self.__record_text(response.chat_completion)

    def snapshot(
        self,
        response: ChatResponse,
        expose_tool_call_requests: bool = False,
    ) -> ChatResponse:
        """Return a response decorated with the assembled visible content parts."""
        self.record_response(response)
        return ChatResponse(
            error=response.error,
            chat_completion=self.__visible_chat_completion(
                response.chat_completion,
                expose_tool_call_requests=expose_tool_call_requests,
            ),
            content_parts=self.content_parts,
        )

    def __record_text(self, chat_completion: Optional[ChatCompletion]) -> None:
        if (
            not chat_completion
            or chat_completion.sender != ChatCompletionSender.BOT
            or not chat_completion.content
        ):
            return
        text_part = ContentPart(type="text", text=chat_completion.content)
        if self.__current_text_part_index is None:
            self.__current_text_part_index = len(self.__content_parts)
            self.__content_parts.append(text_part)
            return
        self.__content_parts[self.__current_text_part_index] = text_part

    def __record_tool_part(self, event: ContentPartTool) -> None:
        tool_part = ContentPart(type="tool", tool=event)
        if event.tool_call_id in self.__tool_part_indexes_by_id:
            part_index = self.__tool_part_indexes_by_id[event.tool_call_id]
            self.__content_parts[part_index] = tool_part
            return
        self.__tool_part_indexes_by_id[event.tool_call_id] = len(self.__content_parts)
        self.__content_parts.append(tool_part)

    def __visible_chat_completion(
        self,
        chat_completion: Optional[ChatCompletion],
        expose_tool_call_requests: bool,
    ) -> Optional[ChatCompletion]:
        if (
            chat_completion is None
            or expose_tool_call_requests
            or not chat_completion.tool_call_requests
        ):
            return chat_completion
        return ChatCompletion(
            sender=chat_completion.sender,
            content=chat_completion.content,
            image_uri=chat_completion.image_uri,
            function_call_request=chat_completion.function_call_request,
            function_call_response=chat_completion.function_call_response,
            tool_call_responses=chat_completion.tool_call_responses,
            response_replay_items=chat_completion.response_replay_items,
            reasoning_summary=chat_completion.reasoning_summary,
        )
