from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
)

from zav.llm_tracing import Span

from zav.agents_sdk.domain.chat_message import ChatMessage, ChatMessageSender
from zav.agents_sdk.domain.tools import ToolsRegistry


def instrument_execute(
    self: "ChatAgent",
    execute: Callable[[List[ChatMessage]], Awaitable[Optional[ChatMessage]]],
):
    async def wrapper(conversation: List[ChatMessage]) -> Optional[ChatMessage]:
        if self.span:
            self.span.update(attributes={"input": conversation[-1].content})
        response = await execute(conversation)
        if self.span:
            if response:
                self.span.end(
                    attributes={
                        "output": {
                            "content": response.content,
                            "evidences": [e.dict() for e in response.evidences or []],
                            "function_call_request": (
                                response.function_call_request.dict()
                                if response.function_call_request
                                else None
                            ),
                            "function_specs": (
                                response.function_specs.dict()
                                if response.function_specs
                                else None
                            ),
                        }
                    }
                )
            else:
                self.span.end()
        return response

    return wrapper


class ChatAgent(ABC):
    agent_name: ClassVar[str]
    span: Optional[Span] = None
    debug_backend: Optional[Callable[[Any], Any]] = None
    tools_registry: ToolsRegistry = ToolsRegistry()

    def __getattribute__(self, name: str) -> Any:
        if name == "execute":
            return instrument_execute(self, super().__getattribute__(name))
        return super().__getattribute__(name)

    def debug(self, msg: Any) -> Any:
        if self.span:
            self.span.add_event(
                name="debug log", attributes={"input": msg, "level": "DEBUG"}
            )
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

    def __getattribute__(self, name: str) -> Any:
        if name == "execute_streaming":
            # TODO: implement
            return super().__getattribute__(name)
        return super().__getattribute__(name)

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
