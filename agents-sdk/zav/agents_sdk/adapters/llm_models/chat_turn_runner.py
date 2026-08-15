from typing import AsyncIterator, List, Optional, Union

from zav.llm_tracing import Span
from zav.prompt_completion import ChatCompletionClient
from zav.prompt_completion import ChatMessage as PcChatMessage
from zav.prompt_completion import ChatResponse as PcChatResponse
from zav.pydantic_compat import PYDANTIC_V2

from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
    ChatResponse,
    convert_chat_response,
    parse_chat_message,
)
from zav.agents_sdk.adapters.llm_models.chat_request_builder import ChatRequestBuilder
from zav.agents_sdk.adapters.llm_models.chat_turn_request import ChatTurnRequest
from zav.agents_sdk.adapters.llm_models.streaming_response_assembler import (
    StreamingResponseAssembler,
)
from zav.agents_sdk.adapters.llm_models.tool_call_runner import (
    ToolCallResult,
    ToolCallRunner,
)
from zav.agents_sdk.domain.chat_message import (
    ChatMessage,
    ContentPart,
    ContentPartTool,
)
from zav.agents_sdk.domain.tools import ToolsRegistry

MAX_NESTING_FINAL_TURN_INSTRUCTION = (
    "You have reached the maximum number of tool-calling steps allowed for this "
    "turn. Do not request any more tools. Provide the best final answer you can "
    "using only the information already gathered, and clearly note what is "
    "missing or could not be completed."
)

MAX_NESTING_FALLBACK_BOT_MESSAGE = (
    "I've reached the maximum number of tool-calling steps for this turn and "
    "was unable to finish. Please continue or narrow the request."
)


class ChatTurnState:
    """Tracks accumulated LLM/tool completions across chat turns.

    ``__base`` is the seed transcript for this state: the completions that
    started the model/tool loop, either converted from a new user request or
    restored from previously persisted state. ``__base_messages`` keeps the raw
    request messages when the seed came from ``messages`` so the first turn can
    be parsed directly without round-tripping through ``ChatCompletion``.

    ``__continuations`` is the append-only suffix after the base: model
    completions, tool completions, final instructions, final responses, and
    later user-turn input added by ``add_turn``.

    The full restartable transcript is ``__base`` plus ``__continuations``.
    Per-turn controls such as remaining tool budget, tool choice, and function
    inclusion are reset for each new user turn.
    """

    def __init__(self) -> None:
        self.__base: List[ChatCompletion] = []
        self.__base_messages: Optional[List[ChatMessage]] = None
        self.__continuations: List[ChatCompletion] = []
        self.__seeded = False
        self.remaining_tool_steps = 0
        self.tool_choice: Optional[str] = None
        self.include_functions = True

    @property
    def chat_completions(self) -> List[ChatCompletion]:
        return self.__completions()

    def restore(self, completions: List[ChatCompletion]) -> None:
        self.__base = list(completions)
        self.__seeded = True

    def add_turn(self, request: ChatTurnRequest) -> None:
        if self.__seeded:
            self.__continuations.extend(self.__request_completions(request))
        else:
            self.__base_messages = request.messages
            self.__base = self.__request_completions(request)
            self.__seeded = True
        self.__apply_turn_controls(request)

    def pc_messages(self) -> List[PcChatMessage]:
        if not self.__continuations and self.__base_messages is not None:
            return [parse_chat_message(message) for message in self.__base_messages]
        return [parse_chat_message(completion) for completion in self.__completions()]

    def append_tool_turn(
        self,
        model_completion: ChatCompletion,
        tool_completion: ChatCompletion,
    ) -> None:
        self.__continuations.extend([model_completion, tool_completion])
        self.remaining_tool_steps -= 1
        self.include_functions = False
        if self.tool_choice == "required":
            self.tool_choice = "auto"

    def append_final_instruction(self) -> None:
        self.__continuations.append(
            ChatCompletion(
                sender=ChatCompletionSender.DEVELOPER,
                content=MAX_NESTING_FINAL_TURN_INSTRUCTION,
            )
        )
        self.include_functions = False

    def append_final_response(self, completion: Optional[ChatCompletion]) -> None:
        if completion is not None:
            if not (completion.function_call_request or completion.tool_call_requests):
                self.__base = [
                    self.__without_response_replay_items(item) for item in self.__base
                ]
                self.__continuations = [
                    self.__without_response_replay_items(item)
                    for item in self.__continuations
                ]
                completion = self.__without_response_replay_items(completion)
            self.__continuations.append(completion)

    def __apply_turn_controls(self, request: ChatTurnRequest) -> None:
        self.remaining_tool_steps = request.max_nesting_level
        self.tool_choice = request.tool_choice
        self.include_functions = True

    def __completions(self) -> List[ChatCompletion]:
        return self.__base + self.__continuations

    def __request_completions(self, request: ChatTurnRequest) -> List[ChatCompletion]:
        if request.messages is not None:
            return [
                ChatCompletion.from_chat_message(message)
                for message in request.messages
            ]
        return request.completions or []

    @staticmethod
    def __without_response_replay_items(completion: ChatCompletion) -> ChatCompletion:
        if completion.response_replay_items is None:
            return completion
        if PYDANTIC_V2:
            return completion.model_copy(update={"response_replay_items": None})
        return completion.copy(update={"response_replay_items": None})


class ChatTurnRunner:
    """Runs the model-to-tools loop for one chat completion request."""

    def __init__(
        self,
        chat_completion_client: ChatCompletionClient,
        request_builder: ChatRequestBuilder,
        span: Optional[Span] = None,
        state: Optional[ChatTurnState] = None,
    ) -> None:
        self.__chat_completion_client = chat_completion_client
        self.__request_builder = request_builder
        self.__span = span
        self.__state: ChatTurnState = state or ChatTurnState()

    @property
    def state(self) -> ChatTurnState:
        return self.__state

    def restore(self, completions: List[ChatCompletion]) -> None:
        self.__state.restore(completions)

    async def run(self, request: ChatTurnRequest) -> ChatResponse:
        if request.max_nesting_level == 0:
            return self.__fallback_response(request)

        state = self.__state_for_request(request)
        while True:
            response = convert_chat_response(
                await self.__complete_once(
                    request=request,
                    state=state,
                    tools=request.tools,
                    tool_choice=state.tool_choice,
                    stream=False,
                )
            )
            if not self.__should_execute_tools(request, response):
                state.append_final_response(response.chat_completion)
                return response

            tool_result = await self.__run_tools(request, response, state.tool_choice)
            if tool_result.should_return_to_user:
                state.append_final_response(tool_result.chat_completion)
                return ChatResponse(chat_completion=tool_result.chat_completion)
            if not (response.chat_completion and tool_result.chat_completion):
                state.append_final_response(response.chat_completion)
                return response

            state.append_tool_turn(
                response.chat_completion, tool_result.chat_completion
            )
            if state.remaining_tool_steps <= 0:
                return await self.__run_final_turn(request, state)

    async def run_streaming(
        self, request: ChatTurnRequest
    ) -> AsyncIterator[ChatResponse]:
        if request.max_nesting_level == 0:
            yield self.__fallback_response(request)
            return

        state = self.__state_for_request(request)
        response_assembler = StreamingResponseAssembler()
        while True:
            should_continue = False
            last_response: Optional[ChatResponse] = None
            chat_response_stream = await self.__complete_once(
                request=request,
                state=state,
                tools=request.tools,
                tool_choice=state.tool_choice,
                stream=True,
            )

            async for chat_response_chunk in chat_response_stream:
                response = convert_chat_response(chat_response_chunk)
                last_response = response
                if request.stream_tool_calls:
                    yield response_assembler.snapshot(
                        response,
                        expose_tool_call_requests=True,
                    )

                if not self.__should_execute_streaming_tools(request, response):
                    if not request.stream_tool_calls:
                        if request.preserve_streamed_content_parts:
                            yield response_assembler.snapshot(response)
                        else:
                            yield ChatResponse(
                                chat_completion=response.chat_completion,
                                content_parts=response_assembler.tool_content_parts,
                            )
                    continue

                if request.preserve_streamed_content_parts:
                    response_assembler.record_response(response)
                tool_result = None
                if request.should_stream_tool_progress:
                    async for event in self.__stream_tool_progress(
                        request, response, state.tool_choice
                    ):
                        if isinstance(event, ContentPartTool):
                            response_assembler.record_tool_event(event)
                            if request.preserve_streamed_content_parts:
                                yield response_assembler.snapshot(response)
                            else:
                                yield ChatResponse(
                                    content_parts=response_assembler.tool_content_parts,
                                )
                        else:
                            tool_result = event
                else:
                    tool_result = await self.__run_tools(
                        request, response, state.tool_choice
                    )

                if tool_result is None:
                    continue

                if self.__should_yield_tool_completion(
                    request, response_assembler, tool_result
                ):
                    if request.preserve_streamed_content_parts:
                        response_assembler.finish_text_segment()
                        yield response_assembler.snapshot(
                            ChatResponse(chat_completion=tool_result.chat_completion),
                        )
                    else:
                        yield ChatResponse(
                            chat_completion=tool_result.chat_completion,
                            content_parts=self.__default_tool_completion_content_parts(
                                tool_result,
                                response_assembler,
                            ),
                        )

                if tool_result.should_return_to_user:
                    state.append_final_response(tool_result.chat_completion)
                    await self.__drain_stream(chat_response_stream)
                    return
                if not (response.chat_completion and tool_result.chat_completion):
                    continue

                state.append_tool_turn(
                    response.chat_completion, tool_result.chat_completion
                )
                self.__finish_tool_turn_content(request, response_assembler)
                await self.__drain_stream(chat_response_stream)
                if state.remaining_tool_steps <= 0:
                    async for final_response in self.__stream_final_turn(
                        request, state, response_assembler
                    ):
                        yield final_response
                    return
                should_continue = True
                break

            if not should_continue:
                state.append_final_response(
                    last_response.chat_completion if last_response else None
                )
                return

    def __state_for_request(self, request: ChatTurnRequest) -> ChatTurnState:
        self.__state.add_turn(request)
        return self.__state

    async def __complete_once(
        self,
        request: ChatTurnRequest,
        state: ChatTurnState,
        tools,
        tool_choice: Optional[str],
        stream: bool,
    ) -> Union[AsyncIterator[PcChatResponse], PcChatResponse]:
        chat_request = await self.__request_builder.build(
            request=request,
            pc_messages=state.pc_messages(),
            tools=tools,
            tool_choice=tool_choice,
            include_functions=state.include_functions,
        )
        return await self.__chat_completion_client.complete(
            request=chat_request, stream=stream
        )

    async def __drain_stream(self, stream: AsyncIterator[PcChatResponse]) -> None:
        async for _ in stream:
            pass

    async def __run_final_turn(
        self, request: ChatTurnRequest, state: ChatTurnState
    ) -> ChatResponse:
        if request.log_fn:
            request.log_fn({"max_nesting_level_reached": True})
        state.append_final_instruction()
        chat_response = await self.__complete_once(
            request=request,
            state=state,
            tools=None,
            tool_choice=None,
            stream=False,
        )
        response = convert_chat_response(chat_response)
        state.append_final_response(response.chat_completion)
        return response

    async def __stream_final_turn(
        self,
        request: ChatTurnRequest,
        state: ChatTurnState,
        response_assembler: StreamingResponseAssembler,
    ) -> AsyncIterator[ChatResponse]:
        if request.log_fn:
            request.log_fn({"max_nesting_level_reached": True})
        state.append_final_instruction()
        chat_response_stream = await self.__complete_once(
            request=request,
            state=state,
            tools=None,
            tool_choice=None,
            stream=True,
        )
        last_response: Optional[ChatResponse] = None
        async for chat_response_chunk in chat_response_stream:
            response = convert_chat_response(chat_response_chunk)
            last_response = response
            if request.preserve_streamed_content_parts:
                yield response_assembler.snapshot(response)
            else:
                yield ChatResponse(
                    chat_completion=response.chat_completion,
                    content_parts=response_assembler.tool_content_parts,
                )
        state.append_final_response(
            last_response.chat_completion if last_response else None
        )

    def __fallback_response(self, request: ChatTurnRequest) -> ChatResponse:
        if request.log_fn:
            request.log_fn({"max_nesting_level_exceeded": True})
        return ChatResponse(
            chat_completion=ChatCompletion(
                sender=ChatCompletionSender.BOT,
                content=MAX_NESTING_FALLBACK_BOT_MESSAGE,
            ),
        )

    def __should_execute_tools(
        self, request: ChatTurnRequest, response: ChatResponse
    ) -> bool:
        return bool(
            response.chat_completion
            and request.execute_tools
            and isinstance(request.tools, ToolsRegistry)
        )

    def __should_execute_streaming_tools(
        self, request: ChatTurnRequest, response: ChatResponse
    ) -> bool:
        return bool(
            self.__should_execute_tools(request, response)
            and response.chat_completion
            and response.chat_completion.tool_call_requests
        )

    async def __run_tools(
        self,
        request: ChatTurnRequest,
        response: ChatResponse,
        tool_choice: Optional[str],
    ) -> ToolCallResult:
        tools_registry = request.tools
        if not isinstance(tools_registry, ToolsRegistry):
            return ToolCallResult()
        runner = self.__tool_runner(tools_registry, request, tool_choice)
        return await runner.execute(
            response.chat_completion.tool_call_requests
            if response.chat_completion
            else None
        )

    async def __stream_tool_progress(
        self,
        request: ChatTurnRequest,
        response: ChatResponse,
        tool_choice: Optional[str],
    ) -> AsyncIterator[Union[ToolCallResult, ContentPartTool]]:
        tools_registry = request.tools
        if not isinstance(tools_registry, ToolsRegistry):
            yield ToolCallResult()
            return
        runner = self.__tool_runner(tools_registry, request, tool_choice)
        async for event in runner.execute_streaming(
            response.chat_completion.tool_call_requests
            if response.chat_completion
            else None
        ):
            yield event

    def __tool_runner(
        self,
        tools_registry: ToolsRegistry,
        request: ChatTurnRequest,
        tool_choice: Optional[str],
    ) -> ToolCallRunner:
        return ToolCallRunner(
            tools_registry=tools_registry,
            tool_choice=tool_choice,
            log_fn=request.log_fn,
            span=self.__span,
            concurrent_tool_execution=request.concurrent_tool_execution,
        )

    def __should_yield_tool_completion(
        self,
        request: ChatTurnRequest,
        response_assembler: StreamingResponseAssembler,
        tool_result: ToolCallResult,
    ) -> bool:
        if not tool_result.chat_completion:
            return False
        if tool_result.should_return_to_user:
            return True
        return bool(request.stream_tool_calls and response_assembler.has_tool_parts)

    def __finish_tool_turn_content(
        self,
        request: ChatTurnRequest,
        response_assembler: StreamingResponseAssembler,
    ) -> None:
        if request.preserve_streamed_content_parts:
            response_assembler.finish_text_segment()
            return
        response_assembler.reset_text()

    def __default_tool_completion_content_parts(
        self,
        tool_result: ToolCallResult,
        response_assembler: StreamingResponseAssembler,
    ) -> List[ContentPart]:
        content_parts = response_assembler.tool_content_parts
        chat_completion = tool_result.chat_completion
        if (
            tool_result.should_return_to_user
            and chat_completion is not None
            and chat_completion.sender == ChatCompletionSender.BOT
            and chat_completion.content
        ):
            return content_parts + [
                ContentPart(type="text", text=chat_completion.content)
            ]
        return content_parts
