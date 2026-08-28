import json
import warnings
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    overload,
)

from typing_extensions import Literal
from zav.llm_tracing import Span
from zav.prompt_completion import ChatCompletionClient

from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
    ChatResponse,
    FunctionCallResponse,
    ToolCallRequest,
    ToolCallResponse,
    parse_chat_message,
)
from zav.agents_sdk.adapters.llm_models.chat_request_builder import ChatRequestBuilder
from zav.agents_sdk.adapters.llm_models.chat_turn_request import (
    STREAM_TOOL_EVENTS_DEPRECATION_MESSAGE,
    ChatTurnRequest,
)
from zav.agents_sdk.adapters.llm_models.chat_turn_runner import (
    MAX_NESTING_FALLBACK_BOT_MESSAGE,
    MAX_NESTING_FINAL_TURN_INSTRUCTION,
    ChatTurnRunner,
)
from zav.agents_sdk.adapters.llm_models.context_window_manager import (
    ContextWindowManager,
)
from zav.agents_sdk.adapters.policies.llm_selection import LLMSelectionConfiguration
from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyFactory,
    ResumableAgentDependency,
)
from zav.agents_sdk.domain.chat_message import ChatMessage
from zav.agents_sdk.domain.llm_client_factory import LLMClientFactory, LLMNotConfigured
from zav.agents_sdk.domain.tools import ToolsRegistry

__all__ = [
    "ChatCompletion",
    "ChatCompletionSender",
    "ChatResponse",
    "FunctionCallResponse",
    "MAX_NESTING_FALLBACK_BOT_MESSAGE",
    "MAX_NESTING_FINAL_TURN_INSTRUCTION",
    "ToolCallRequest",
    "ToolCallResponse",
    "ResumableZAVChatCompletionClient",
    "ResumableZAVChatCompletionClientFactory",
    "ZAVChatCompletionClient",
    "ZAVChatCompletionClientFactory",
    "parse_chat_message",
]


class ZAVChatCompletionClient:
    def __init__(
        self,
        chat_completion_client: ChatCompletionClient,
        context_window_manager: ContextWindowManager,
        max_nesting_level: int,
        span: Optional[Span] = None,
    ) -> None:
        self.__chat_completion_client = chat_completion_client
        self.__span = span
        self.__context_window_manager = context_window_manager
        self.__max_nesting_level = max_nesting_level

    @overload
    async def complete(  # type: ignore
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: Optional[int] = None,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Literal[False] = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        stream_tool_events: Optional[bool] = None,
        stream_tool_progress: bool = False,
        preserve_streamed_content_parts: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        logprobs: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        concurrent_tool_execution: bool = False,
        seed: Optional[int] = None,
        verbosity: Optional[str] = None,
    ) -> ChatResponse:
        pass

    @overload
    async def complete(
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: Optional[int] = None,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Literal[True] = True,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        stream_tool_events: Optional[bool] = None,
        stream_tool_progress: bool = False,
        preserve_streamed_content_parts: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        logprobs: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        concurrent_tool_execution: bool = False,
        seed: Optional[int] = None,
        verbosity: Optional[str] = None,
    ) -> AsyncIterator[ChatResponse]:
        pass

    @overload
    async def complete(
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: Optional[int] = None,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        stream_tool_events: Optional[bool] = None,
        stream_tool_progress: bool = False,
        preserve_streamed_content_parts: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        logprobs: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        concurrent_tool_execution: bool = False,
        seed: Optional[int] = None,
        verbosity: Optional[str] = None,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]:
        pass

    async def complete(
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: Optional[int] = None,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Union[Literal[True, False], bool] = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        stream_tool_events: Optional[bool] = None,
        stream_tool_progress: bool = False,
        preserve_streamed_content_parts: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        logprobs: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        concurrent_tool_execution: bool = False,
        seed: Optional[int] = None,
        verbosity: Optional[str] = None,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]:
        request = self.build_request(
            messages=messages,
            completions=completions,
            max_tokens=max_tokens,
            bot_setup_description=bot_setup_description,
            functions=functions,
            tools=tools,
            tool_choice=tool_choice,
            execute_tools=execute_tools,
            stream_tool_calls=stream_tool_calls,
            stream_tool_events=stream_tool_events,
            stream_tool_progress=stream_tool_progress,
            preserve_streamed_content_parts=preserve_streamed_content_parts,
            log_fn=log_fn,
            max_nesting_level=max_nesting_level,
            reasoning_effort=reasoning_effort,
            logprobs=logprobs,
            parallel_tool_calls=parallel_tool_calls,
            concurrent_tool_execution=concurrent_tool_execution,
            seed=seed,
            verbosity=verbosity,
        )
        runner = ChatTurnRunner(
            chat_completion_client=self.__chat_completion_client,
            request_builder=ChatRequestBuilder(self.__context_window_manager),
            span=self.__span,
        )
        if stream:
            return runner.run_streaming(request)
        return await runner.run(request)

    def build_request(
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: Optional[int] = None,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        stream_tool_events: Optional[bool] = None,
        stream_tool_progress: bool = False,
        preserve_streamed_content_parts: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        logprobs: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        concurrent_tool_execution: bool = False,
        seed: Optional[int] = None,
        verbosity: Optional[str] = None,
    ) -> ChatTurnRequest:
        return ChatTurnRequest(
            messages=messages,
            completions=completions,
            max_tokens=max_tokens,
            bot_setup_description=bot_setup_description,
            functions=functions,
            tools=tools,
            tool_choice=tool_choice,
            execute_tools=execute_tools,
            stream_tool_calls=stream_tool_calls,
            stream_tool_progress=self.__resolve_stream_tool_progress(
                stream_tool_progress,
                stream_tool_events,
            ),
            preserve_streamed_content_parts=preserve_streamed_content_parts,
            log_fn=log_fn,
            max_nesting_level=(
                self.__max_nesting_level
                if max_nesting_level is None
                else max_nesting_level
            ),
            reasoning_effort=reasoning_effort,
            logprobs=logprobs,
            parallel_tool_calls=parallel_tool_calls,
            concurrent_tool_execution=concurrent_tool_execution,
            seed=seed,
            verbosity=verbosity,
        )

    def __resolve_stream_tool_progress(
        self,
        stream_tool_progress: bool,
        stream_tool_events: Optional[bool],
    ) -> bool:
        if stream_tool_events is None:
            return stream_tool_progress
        warnings.warn(
            STREAM_TOOL_EVENTS_DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=3,
        )
        return stream_tool_progress or stream_tool_events


class ResumableZAVChatCompletionClient(ResumableAgentDependency):
    state_key = "zav_chat_completion_client"
    # Its restored state is the model transcript (the runner's chat_completions),
    # i.e. the conversation itself — so a resumed turn needs only the new message,
    # and the handler must not re-feed it the full stored transcript.
    restores_conversation = True

    def __init__(self, client: ZAVChatCompletionClient, runner: ChatTurnRunner) -> None:
        self.__client = client
        self.__runner = runner
        self.__dumped_completion_count: Optional[int] = None
        self.__dumped_state: Optional[Dict[str, Any]] = None

    async def dump(self) -> Dict[str, Any]:
        # The transcript is append-only within a turn, so the completion count is a
        # faithful change signal: an unchanged count means unchanged content. Reuse
        # the previous serialization instead of re-serializing on every checkpoint.
        completions = self.__runner.state.chat_completions
        if self.__dumped_state is not None and self.__dumped_completion_count == len(
            completions
        ):
            return self.__dumped_state
        self.__dumped_state = {
            "chat_completions": [
                json.loads(chat_completion.json()) for chat_completion in completions
            ]
        }
        self.__dumped_completion_count = len(completions)
        return self.__dumped_state

    async def load(self, state: Dict[str, Any]) -> None:
        self.__runner.restore(
            [
                ChatCompletion.parse_obj(chat_completion_state)
                for chat_completion_state in state.get("chat_completions", [])
            ]
        )
        self.__dumped_completion_count = None
        self.__dumped_state = None

    @property
    def is_resumed(self) -> bool:
        # True once a prior transcript has been restored — i.e. this is a
        # continued/resumed stateful turn rather than the first turn. Lets a
        # caller seed turn-invariant context (e.g. the global conversation
        # context) into the transcript only on the first turn instead of
        # re-appending it on every turn.
        return len(self.__runner.state.chat_completions) > 0

    async def complete(
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: Optional[int] = None,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Union[Literal[True, False], bool] = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        stream_tool_events: Optional[bool] = None,
        stream_tool_progress: bool = False,
        preserve_streamed_content_parts: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        logprobs: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        concurrent_tool_execution: bool = False,
        seed: Optional[int] = None,
        verbosity: Optional[str] = None,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]:
        request = self.__client.build_request(
            messages=messages,
            completions=completions,
            max_tokens=max_tokens,
            bot_setup_description=bot_setup_description,
            functions=functions,
            tools=tools,
            tool_choice=tool_choice,
            execute_tools=execute_tools,
            stream_tool_calls=stream_tool_calls,
            stream_tool_events=stream_tool_events,
            stream_tool_progress=stream_tool_progress,
            preserve_streamed_content_parts=preserve_streamed_content_parts,
            log_fn=log_fn,
            max_nesting_level=max_nesting_level,
            reasoning_effort=reasoning_effort,
            logprobs=logprobs,
            parallel_tool_calls=parallel_tool_calls,
            concurrent_tool_execution=concurrent_tool_execution,
            seed=seed,
            verbosity=verbosity,
        )
        if stream:
            return self.__runner.run_streaming(request)
        return await self.__runner.run(request)


class _DeferredChatCompletionClient(ChatCompletionClient):
    """Resolves the LLM selection on first use, so building the ZAV client
    never requires a resolvable selection (tool listing must succeed with no
    LLM in scope)."""

    def __init__(self, builder: Callable[[], ChatCompletionClient]):
        self.__builder = builder
        self.__client: Optional[ChatCompletionClient] = None

    @classmethod
    def from_configuration(
        cls,
        vendor_configuration,
        model_configuration,
        span: Optional[Span] = None,
    ) -> ChatCompletionClient:
        raise NotImplementedError

    async def complete(self, request, stream=False):
        if self.__client is None:
            self.__client = self.__builder()
        return await self.__client.complete(request, stream=stream)


def _deferred_chat_completion_client(
    llm_client_factory: Optional[LLMClientFactory],
    llm_selection_configuration: LLMSelectionConfiguration,
    span: Optional[Span],
) -> ChatCompletionClient:
    def build() -> ChatCompletionClient:
        if llm_client_factory is None:
            raise LLMNotConfigured("No LLM client factory in scope.")
        return llm_client_factory.create_chat_completion_client(
            requested=llm_selection_configuration.llm_configuration_name,
            span=span,
        )

    return _DeferredChatCompletionClient(build)


class ZAVChatCompletionClientFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        context_window_manager: ContextWindowManager,
        llm_client_factory: Optional[LLMClientFactory] = None,
        llm_selection_configuration: LLMSelectionConfiguration = (
            LLMSelectionConfiguration()
        ),
        max_nesting_level: int = 20,
        span: Optional[Span] = None,
    ) -> ZAVChatCompletionClient:
        chat_completion_client = _deferred_chat_completion_client(
            llm_client_factory, llm_selection_configuration, span
        )
        return ZAVChatCompletionClient(
            chat_completion_client,
            context_window_manager=context_window_manager,
            max_nesting_level=max_nesting_level,
            span=span,
        )


class ResumableZAVChatCompletionClientFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        context_window_manager: ContextWindowManager,
        llm_client_factory: Optional[LLMClientFactory] = None,
        llm_selection_configuration: LLMSelectionConfiguration = (
            LLMSelectionConfiguration()
        ),
        max_nesting_level: int = 20,
        span: Optional[Span] = None,
    ) -> ResumableZAVChatCompletionClient:
        chat_completion_client = _deferred_chat_completion_client(
            llm_client_factory, llm_selection_configuration, span
        )
        client = ZAVChatCompletionClient(
            chat_completion_client,
            context_window_manager=context_window_manager,
            max_nesting_level=max_nesting_level,
            span=span,
        )
        runner = ChatTurnRunner(
            chat_completion_client=chat_completion_client,
            request_builder=ChatRequestBuilder(context_window_manager),
            span=span,
        )
        return ResumableZAVChatCompletionClient(client, runner)
