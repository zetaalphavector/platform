from typing import AsyncIterator, List, Optional, Set, Union

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
from zav.agents_sdk.adapters.llm_models.chat_turn_request import (
    ChatTurnRequest,
    InnerLoopContext,
)
from zav.agents_sdk.adapters.llm_models.queued_messages import QueuedMessages
from zav.agents_sdk.adapters.llm_models.streaming_response_assembler import (
    StreamingResponseAssembler,
)
from zav.agents_sdk.adapters.llm_models.tool_call_runner import (
    ToolCallResult,
    ToolCallRunner,
)
from zav.agents_sdk.adapters.llm_models.turn_inbox import ToolApprovals
from zav.agents_sdk.adapters.tool_confirmation.session_approvals import (
    SessionApprovals,
)
from zav.agents_sdk.adapters.tool_policy import (
    ToolPolicyProvider,
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
    ``turn_starts`` holds, per user message, the transcript position where its
    input begins, so a rewound conversation can cut the transcript at a message.
    Per-turn controls such as remaining tool budget, tool choice, and function
    inclusion are reset for each new user turn.
    """

    def __init__(self) -> None:
        self.__base: List[ChatCompletion] = []
        self.__base_messages: Optional[List[ChatMessage]] = None
        self.__continuations: List[ChatCompletion] = []
        self.__turn_starts: List[int] = []
        self.__seeded = False
        self.remaining_tool_steps = 0
        self.tool_choice: Optional[str] = None
        self.include_functions = True

    @property
    def chat_completions(self) -> List[ChatCompletion]:
        return self.__completions()

    @property
    def turn_starts(self) -> List[int]:
        return list(self.__turn_starts)

    def restore(
        self, completions: List[ChatCompletion], turn_starts: Optional[List[int]] = None
    ) -> None:
        self.__base = list(completions)
        self.__turn_starts = list(turn_starts or [])
        self.__seeded = True

    def add_turn(self, request: ChatTurnRequest) -> None:
        turn_input = self.__request_completions(request)
        self.__record_turn_starts(turn_input)
        if self.__seeded:
            self.__continuations.extend(turn_input)
        else:
            self.__base_messages = request.messages
            self.__base = turn_input
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

    def append_interjections(self, completions: List[ChatCompletion]) -> None:
        self.__continuations.extend(completions)

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

    def __record_turn_starts(self, turn_input: List[ChatCompletion]) -> None:
        # One start per user message, as the stored conversation counts them: a
        # turn can start with several. A message's start takes in the context
        # completions just before it. A turn resumed from its checkpoint carries
        # no user message.
        offset = len(self.__base) + len(self.__continuations)
        start: Optional[int] = None
        for index, completion in enumerate(turn_input):
            if start is None:
                start = index
            if completion.sender == ChatCompletionSender.USER:
                self.__turn_starts.append(offset + start)
                start = None
            elif completion.sender != ChatCompletionSender.DEVELOPER:
                start = None

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
    """Runs the model-to-tools loop for one chat completion request.

    With queued messages (the root agent of an interactive turn whose agent
    reads them), the messages the user sends while the turn runs are added to
    the transcript before each model call, and a turn about to end continues
    while more wait."""

    def __init__(
        self,
        chat_completion_client: ChatCompletionClient,
        request_builder: ChatRequestBuilder,
        span: Optional[Span] = None,
        state: Optional[ChatTurnState] = None,
        queued_messages: Optional[QueuedMessages] = None,
        approvals: Optional[ToolApprovals] = None,
        tool_policy: Optional[ToolPolicyProvider] = None,
        session_approvals: Optional[SessionApprovals] = None,
    ) -> None:
        self.__chat_completion_client = chat_completion_client
        self.__request_builder = request_builder
        self.__span = span
        self.__state: ChatTurnState = state or ChatTurnState()
        self.__tool_policy = tool_policy or ToolPolicyProvider([])
        self.__session_approvals = session_approvals
        self.__queued_messages = queued_messages
        self.__approvals = approvals

    @property
    def state(self) -> ChatTurnState:
        return self.__state

    def restore(
        self, completions: List[ChatCompletion], turn_starts: Optional[List[int]] = None
    ) -> None:
        self.__state.restore(completions, turn_starts)

    async def run(self, request: ChatTurnRequest) -> ChatResponse:
        answer = ChatResponse()
        async for answer in self.__turn(request, streaming=False):
            pass
        return answer

    async def run_streaming(
        self, request: ChatTurnRequest
    ) -> AsyncIterator[ChatResponse]:
        async for response in self.__turn(request, streaming=True):
            yield response

    async def __turn(
        self, request: ChatTurnRequest, streaming: bool
    ) -> AsyncIterator[ChatResponse]:
        """The turn: the model answers; while it calls tools, they run and it
        answers again. Messages the user sends meanwhile are read before each
        model call. ``_TurnOutput`` decides what the caller sees of it."""
        if request.max_nesting_level == 0:
            yield self.__fallback_response(request)
            return
        state = self.__state_for_request(request)
        out = _TurnOutput(request, streaming)
        while True:
            for shown in out.user_messages(
                await self.__read_new_messages(request, state)
            ):
                yield shown
            tools = await self.__tools_for_round(request, state)

            answer: Optional[ChatResponse] = None
            tool_result: Optional[ToolCallResult] = None
            model_stream = self.__ask_model(request, state, streaming, tools)
            async for answer in model_stream:
                calls = self.__tool_calls(request, answer)
                for shown in out.model_chunk(answer, runs_tools=bool(calls)):
                    yield shown
                if not calls:
                    continue
                async for event in self.__run_calls(
                    request, state, tools, calls, streaming
                ):
                    if isinstance(event, ContentPartTool):
                        yield out.tool_event(answer, event)
                    else:
                        tool_result = event
                if tool_result is not None:
                    break
            await self.__drain(model_stream)

            if tool_result is None and answer is not None:
                tool_result = await self.__required_tool_reminder(
                    request, state, tools, answer
                )
            if tool_result is not None and tool_result.chat_completion is not None:
                for shown in out.tool_result(tool_result):
                    yield shown
                if tool_result.should_return_to_user:
                    await self.__end(
                        request, state, answer, tool_result.chat_completion
                    )
                    for shown in out.answer(
                        ChatResponse(chat_completion=tool_result.chat_completion)
                    ):
                        yield shown
                    return
                if answer is not None and answer.chat_completion is not None:
                    state.append_tool_turn(
                        answer.chat_completion, tool_result.chat_completion
                    )
                    state.append_interjections(
                        await self.__after_model_call(request, state, answer)
                    )
                    out.next_round()
                    if state.remaining_tool_steps <= 0:
                        async for shown in self.__final_answer(request, state, out):
                            yield shown
                        return
                    continue

            await self.__end(
                request, state, answer, answer.chat_completion if answer else None
            )
            # A message that arrived while the model wrote its answer is
            # answered in this turn rather than left for the next one.
            if (
                self.__queued_messages is not None
                and await self.__queued_messages.has_messages()
            ):
                out.next_round()
                continue
            if answer is not None:
                for shown in out.answer(answer):
                    yield shown
            return

    def __state_for_request(self, request: ChatTurnRequest) -> ChatTurnState:
        self.__state.add_turn(request)
        return self.__state

    async def __read_new_messages(
        self, request: ChatTurnRequest, state: ChatTurnState
    ) -> List[ChatMessage]:
        """Adds to the transcript what arrived since the last model call: the
        user's queued messages, then the hooks' interjections. Returns the
        user's messages."""
        delivered = await self.__deliver_queued_messages(state)
        if request.hooks is not None:
            interjections = await request.hooks.before_model_call(
                InnerLoopContext(
                    transcript=state.chat_completions,
                    remaining_tool_steps=state.remaining_tool_steps,
                )
            )
            if interjections:
                state.append_interjections(interjections)
        return delivered

    async def __deliver_queued_messages(
        self, state: ChatTurnState
    ) -> List[ChatMessage]:
        if self.__queued_messages is None:
            return []
        messages = await self.__queued_messages.take()
        if not messages:
            return []
        state.append_interjections(await self.__queued_messages.frame(messages))
        return messages

    async def __after_model_call(
        self, request: ChatTurnRequest, state: ChatTurnState, response: ChatResponse
    ) -> List[ChatCompletion]:
        """Runs once the model's response and, if it called tools, their results
        are in the transcript. Returns the messages the hooks want appended;
        empty when there are no hooks."""
        if request.hooks is None or response.chat_completion is None:
            return []
        return await request.hooks.after_model_call(
            InnerLoopContext(
                transcript=state.chat_completions,
                remaining_tool_steps=state.remaining_tool_steps,
                response=response.chat_completion,
            )
        )

    async def __complete_once(
        self,
        request: ChatTurnRequest,
        state: ChatTurnState,
        tools,
        tool_choice: Optional[str],
        on_demand: Set[str],
        stream: bool,
    ) -> Union[AsyncIterator[PcChatResponse], PcChatResponse]:
        chat_request = await self.__request_builder.build(
            request=request,
            pc_messages=state.pc_messages(),
            tools=tools,
            tool_choice=tool_choice,
            include_functions=state.include_functions,
            on_demand=on_demand,
        )
        return await self.__chat_completion_client.complete(
            request=chat_request, stream=stream
        )

    async def __ask_model(
        self,
        request: ChatTurnRequest,
        state: ChatTurnState,
        streaming: bool,
        tools: Optional[ToolsRegistry],
        final: bool = False,
    ) -> AsyncIterator[ChatResponse]:
        """The model's answer as it grows when streaming, else in one piece.
        It is offered this round's tools it may call directly; the final
        answer, none."""
        on_demand: Set[str] = set()
        if final:
            offered, tool_choice = None, None
        elif tools is None:
            offered, tool_choice = request.tools, state.tool_choice
        else:
            offered = tools.for_caller("direct")
            tool_choice = state.tool_choice
            on_demand = {
                name
                for name, tool in tools.tools_index.items()
                if tool.options.on_demand
            }
        if not streaming:
            yield convert_chat_response(
                await self.__complete_once(
                    request, state, offered, tool_choice, on_demand, False
                )
            )
            return
        chunks = await self.__complete_once(
            request, state, offered, tool_choice, on_demand, True
        )
        async for chunk in chunks:
            yield convert_chat_response(chunk)

    @staticmethod
    async def __drain(model_stream: AsyncIterator[ChatResponse]) -> None:
        async for _ in model_stream:
            pass

    async def __end(
        self,
        request: ChatTurnRequest,
        state: ChatTurnState,
        answer: Optional[ChatResponse],
        completion: Optional[ChatCompletion],
    ) -> None:
        state.append_final_response(completion)
        if answer is not None:
            state.append_interjections(
                await self.__after_model_call(request, state, answer)
            )

    async def __final_answer(
        self, request: ChatTurnRequest, state: ChatTurnState, out: "_TurnOutput"
    ) -> AsyncIterator[ChatResponse]:
        """Out of tool steps: one more model call, without tools."""
        if request.log_fn:
            request.log_fn({"max_nesting_level_reached": True})
        await self.__read_new_messages(request, state)
        state.append_final_instruction()
        answer: Optional[ChatResponse] = None
        async for answer in self.__ask_model(
            request, state, out.streaming, tools=None, final=True
        ):
            for shown in out.final_chunk(answer):
                yield shown
        state.append_final_response(answer.chat_completion if answer else None)
        if answer is not None:
            for shown in out.answer(answer):
                yield shown

    def __fallback_response(self, request: ChatTurnRequest) -> ChatResponse:
        if request.log_fn:
            request.log_fn({"max_nesting_level_exceeded": True})
        return ChatResponse(
            chat_completion=ChatCompletion(
                sender=ChatCompletionSender.BOT,
                content=MAX_NESTING_FALLBACK_BOT_MESSAGE,
            ),
        )

    @staticmethod
    def __tool_calls(request: ChatTurnRequest, answer: ChatResponse):
        """The calls of an answer the turn runs: none when the caller runs
        tools itself or has no registry."""
        if not (
            answer.chat_completion
            and request.execute_tools
            and isinstance(request.tools, ToolsRegistry)
        ):
            return []
        return answer.chat_completion.tool_call_requests or []

    async def __tools_for_round(
        self, request: ChatTurnRequest, state: ChatTurnState
    ) -> Optional[ToolsRegistry]:
        """This round's tools: the request's tools as they are now (its
        providers' listings are cached reads), the ones the tool rules keep,
        with the options they set, plus the context window's
        retrieval tools. A tool not kept is neither offered nor runnable: a
        call to it, from history or invented, gets "not found". None when the
        request has no registry."""
        if not isinstance(request.tools, ToolsRegistry):
            return None
        offered = await self.__tool_policy.apply(
            await request.tools.current_tools(), state.chat_completions
        )
        return request.tools.offering(
            offered.tools + self.__request_builder.retrieval_tools()
        )

    async def __run_calls(
        self,
        request: ChatTurnRequest,
        state: ChatTurnState,
        tools: Optional[ToolsRegistry],
        calls,
        streaming: bool,
    ) -> AsyncIterator[Union[ContentPartTool, ToolCallResult]]:
        """Runs the calls through this round's registry: a tool not in it gets
        "not found". Yields each tool's progress when the caller streams it.
        """
        if tools is None:
            yield ToolCallResult()
            return
        runner = self.__tool_runner(tools, request, state.tool_choice)
        if streaming and request.should_stream_tool_progress:
            async for event in runner.execute_streaming(calls):
                yield event
            return
        yield await runner.execute(calls)

    async def __required_tool_reminder(
        self,
        request: ChatTurnRequest,
        state: ChatTurnState,
        tools: Optional[ToolsRegistry],
        answer: ChatResponse,
    ) -> Optional[ToolCallResult]:
        """When ``tool_choice`` requires a tool and the answer called none,
        the reminder the runner adds so the model calls one."""
        if (
            state.tool_choice in (None, "auto", "none")
            or tools is None
            or not request.execute_tools
            or answer.chat_completion is None
        ):
            return None
        runner = self.__tool_runner(tools, request, state.tool_choice)
        return await runner.execute(None)

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
            session_approvals=self.__session_approvals,
            approvals=self.__approvals,
        )


class _TurnOutput:
    """What the caller sees of a turn. Streaming, each step shows as it
    happens, shaped by the request's display flags; otherwise only the
    answer."""

    def __init__(self, request: ChatTurnRequest, streaming: bool) -> None:
        self.streaming = streaming
        self.__preserve = request.preserve_streamed_content_parts
        self.__stream_tool_calls = request.stream_tool_calls
        self.__assembler = StreamingResponseAssembler()

    def user_messages(self, messages: List[ChatMessage]) -> List[ChatResponse]:
        if not (self.streaming and self.__preserve and messages):
            return []
        for message in messages:
            self.__assembler.record_user_message(message)
        return [self.__assembler.snapshot(ChatResponse())]

    def model_chunk(
        self, response: ChatResponse, runs_tools: bool
    ) -> List[ChatResponse]:
        if not self.streaming:
            return []
        shown = []
        if self.__stream_tool_calls:
            shown.append(
                self.__assembler.snapshot(response, expose_tool_call_requests=True)
            )
        if runs_tools:
            if self.__preserve:
                self.__assembler.record_response(response)
        elif not self.__stream_tool_calls:
            shown.append(self.__text(response))
        return shown

    def tool_event(
        self, response: ChatResponse, event: ContentPartTool
    ) -> ChatResponse:
        self.__assembler.record_tool_event(event)
        if self.__preserve:
            return self.__assembler.snapshot(response)
        return ChatResponse(content_parts=self.__assembler.tool_content_parts)

    def tool_result(self, result: ToolCallResult) -> List[ChatResponse]:
        if not self.streaming or not (
            result.should_return_to_user
            or (self.__stream_tool_calls and self.__assembler.has_tool_parts)
        ):
            return []
        if self.__preserve:
            self.__assembler.finish_text_segment()
            return [
                self.__assembler.snapshot(
                    ChatResponse(chat_completion=result.chat_completion)
                )
            ]
        parts = self.__assembler.tool_content_parts
        completion = result.chat_completion
        if (
            result.should_return_to_user
            and completion is not None
            and completion.sender == ChatCompletionSender.BOT
            and completion.content
        ):
            parts = parts + [ContentPart(type="text", text=completion.content)]
        return [ChatResponse(chat_completion=completion, content_parts=parts)]

    def final_chunk(self, response: ChatResponse) -> List[ChatResponse]:
        return [self.__text(response)] if self.streaming else []

    def answer(self, response: ChatResponse) -> List[ChatResponse]:
        """The turn's answer, which a streaming caller has already been shown."""
        return [] if self.streaming else [response]

    def next_round(self) -> None:
        if self.__preserve:
            self.__assembler.finish_text_segment()
        else:
            self.__assembler.reset_text()

    def __text(self, response: ChatResponse) -> ChatResponse:
        if self.__preserve:
            return self.__assembler.snapshot(response)
        return ChatResponse(
            error=response.error,
            chat_completion=response.chat_completion,
            content_parts=self.__assembler.tool_content_parts,
        )
