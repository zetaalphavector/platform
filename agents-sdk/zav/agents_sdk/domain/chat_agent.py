from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, ClassVar, Dict, List, Optional, Union

from zav.agents_sdk.domain.chat_message import ChatMessage, ChatMessageSender
from zav.agents_sdk.domain.tools import ToolsRegistry


class ChatAgent(ABC):
    agent_name: ClassVar[str]
    debug_backend: Optional[Callable[[Any], Any]] = None
    tools_registry: ToolsRegistry = ToolsRegistry()

    def debug(self, msg: Any) -> Any:
        if self.debug_backend:
            return self.debug_backend(msg)

    def log_context(
        self,
        llm_contexts: Union[List[str], List[Dict[str, Any]]],
        citation_id_field: Optional[str] = "id",
    ):
        self.debug(
            {
                "llm_contexts": [
                    {
                        **(
                            {"citation_id": citation_id}
                            if citation_id_field
                            and isinstance(context, dict)
                            and (citation_id := context.get(citation_id_field))
                            else {}
                        ),
                        "content": (
                            {k: v for k, v in context.items() if k != citation_id_field}
                            if isinstance(context, dict)
                            else context
                        ),
                    }
                    for context in llm_contexts
                ]
            }
        )

    @abstractmethod
    async def execute(self, conversation: List[ChatMessage]) -> Optional[ChatMessage]:
        raise NotImplementedError


class StreamableChatAgent(ChatAgent):
    async def execute(self, conversation: List[ChatMessage]) -> Optional[ChatMessage]:
        full_message: Optional[ChatMessage] = None
        async for message in self.execute_streaming(conversation):
            full_message = message
        if not full_message:
            return ChatMessage(
                sender=ChatMessageSender.BOT,
                content=(
                    "Apologies, I am unable to assist you with your request"
                    "at the moment. Please contact support if the issue persists."
                ),
            )
        if not full_message.evidences:
            full_message.evidences = None
        return full_message

    @abstractmethod
    def execute_streaming(
        self, conversation: List[ChatMessage]
    ) -> AsyncGenerator[ChatMessage, None]:
        raise NotImplementedError
