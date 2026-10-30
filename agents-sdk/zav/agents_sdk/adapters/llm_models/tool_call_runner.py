import asyncio
import json
from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    ContextManager,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

from zav.llm_tracing import Span
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
    ToolCallRequest,
    ToolCallResponse,
)
from zav.agents_sdk.adapters.llm_models.turn_inbox import ToolApprovals
from zav.agents_sdk.adapters.tool_confirmation.session_approvals import (
    SessionApprovals,
)
from zav.agents_sdk.domain.agent_creator import SubAgentToolResult
from zav.agents_sdk.domain.chat_message import (
    ContentPart,
    ContentPartTool,
    FunctionCallRequest,
)
from zav.agents_sdk.domain.tools import (
    CallDecision,
    ResultDecision,
    Tool,
    ToolsRegistry,
    apply_transform,
    format_display_text,
)
from zav.agents_sdk.domain.turn_inbox import ApprovalOutcome

CONFIRMATION_REQUIRED_DISPLAY_TEXT = "Waiting for your confirmation"
RESULT_BLOCKED_DISPLAY_TEXT = "Blocked"
TOOL_DECLINED_MESSAGE = (
    "The user declined this action. Do not retry it unless the user asks."
)
TOOL_NOT_CONFIRMED_MESSAGE = (
    "The user did not confirm this action and sent a new message instead. "
    "Do not retry it unless the user asks."
)
TOOL_APPROVAL_TIMED_OUT_MESSAGE = (
    "The user did not answer the approval request in time. Do not retry it "
    "unless the user asks."
)
TOOL_CONFIRMATION_UNAVAILABLE_MESSAGE = (
    "This action requires the user's confirmation, which cannot be requested "
    "here. Do not retry it. Report that the action needs the user's confirmation "
    "in the main chat."
)

CallCheck = Callable[
    [ToolCallRequest, List[ToolCallRequest]], Awaitable[Optional[CallDecision]]
]
ResultCheck = Callable[[ToolCallRequest, Any], Awaitable[Optional[ResultDecision]]]


@dataclass(frozen=True)
class ToolCallVerdict:
    """What the runner does with one model-requested call."""

    action: Literal["run", "ask", "decline"]
    params: Optional[Dict[str, Any]] = None
    # Prefix for the tool result so the model describes what actually ran.
    note: Optional[str] = None
    # Tool response for a declined call, and the status the UI shows for it.
    message: Optional[str] = None
    status: Optional[str] = None
    # Why a rule declined the call, for the user; the model gets ``message``.
    rule_reason: Optional[str] = None


CONFIRMATION_DECLINED_DISPLAY_TEXT = "Declined"


class ToolCallResult(BaseModel):
    """Result of executing a batch of model-requested tool calls."""

    chat_completion: Optional[ChatCompletion] = None
    should_return_to_user: bool = False


def _parse_result_dict(exec_response: Any) -> Optional[dict]:
    if isinstance(exec_response, dict):
        return exec_response
    if isinstance(exec_response, str):
        try:
            parsed = json.loads(exec_response)
        except (json.JSONDecodeError, ValueError):
            return None
        # A JSON array or scalar result is text to the model, not a response object.
        return parsed if isinstance(parsed, dict) else None
    return None


def _tool_text(exec_response: Any) -> str:
    if isinstance(exec_response, (dict, list)):
        return json.dumps(exec_response, ensure_ascii=False, default=str)
    return str(exec_response)


class ToolCallRunner:
    """Executes model-requested tools and optional streaming tool events."""

    def __init__(
        self,
        tools_registry: ToolsRegistry,
        tool_choice: Optional[str] = None,
        log_fn: Optional[Callable] = None,
        span: Optional[Span] = None,
        concurrent_tool_execution: bool = False,
        session_approvals: Optional[SessionApprovals] = None,
        approvals: Optional[ToolApprovals] = None,
        check_call: Optional[CallCheck] = None,
        check_result: Optional[ResultCheck] = None,
    ) -> None:
        # Runs tools through the round's whole registry, so a tool that calls
        # others (the REPL) gets them all; the model may call only the ones
        # it may call directly.
        self.__tools_registry = tools_registry
        self.__callable: Dict[str, Tool] = {
            name: tool
            for name, tool in tools_registry.tools_index.items()
            if "direct" in tool.allowed_callers
        }
        self.__approvals = approvals
        self.__tool_choice = tool_choice
        self.__log_fn = log_fn
        self.__span = span
        self.__concurrent_tool_execution = concurrent_tool_execution
        self.__session_approvals = session_approvals
        self.__check_call = check_call
        self.__check_result = check_result
        # A rule's ask holds per call, so its card offers no "allow for this chat".
        self.__rule_asks: Set[str] = set()

    async def execute(
        self, tool_call_requests: Optional[List[ToolCallRequest]] = None
    ) -> ToolCallResult:
        async for event in self.execute_streaming(tool_call_requests):
            if isinstance(event, ToolCallResult):
                return event
        return ToolCallResult()

    async def execute_streaming(  # noqa: C901
        self,
        tool_call_requests: Optional[List[ToolCallRequest]] = None,
    ) -> AsyncIterator[Union[ContentPartTool, ToolCallResult]]:
        if (
            self.__tool_choice
            and self.__tool_choice not in ("auto", "none")
            and not tool_call_requests
        ):
            if self.__log_fn:
                self.__log_fn(
                    {
                        f"Error: tool_choice set to {self.__tool_choice} "
                        "but no tools were called"
                    }
                )
            if self.__tool_choice == "required":
                available_tools = ", ".join(self.__callable.keys())
                yield ToolCallResult(
                    chat_completion=ChatCompletion(
                        sender=ChatCompletionSender.DEVELOPER,
                        content="You must use a tool in your response. "
                        f"The available tools are: {available_tools}",
                    ),
                )
                return

            yield ToolCallResult(
                chat_completion=ChatCompletion(
                    sender=ChatCompletionSender.DEVELOPER,
                    content=f"You must use the '{self.__tool_choice}' tool to respond. "
                    f"Please call '{self.__tool_choice}'.",
                ),
            )
            return

        if not tool_call_requests:
            yield ToolCallResult()
            return

        if self.__log_fn:
            self.__log_fn({"Tool Call Requests": tool_call_requests})

        verdicts = [
            await self.__verdict(tool_call_request, tool_call_requests[:index])
            for index, tool_call_request in enumerate(tool_call_requests)
        ]
        asked = {
            tool_call_request.id
            for tool_call_request, verdict in zip(tool_call_requests, verdicts)
            if verdict.action == "ask"
        }
        if asked:
            async for card in self.__settle_approvals(tool_call_requests, verdicts):
                yield card

        tool_metas: List[Dict[str, Any]] = []
        for tool_call_request, verdict in zip(tool_call_requests, verdicts):
            tool_call_id = tool_call_request.id
            func = tool_call_request.function_call_request
            tool_name = func.name
            tool_params = verdict.params if verdict.action == "run" else func.params

            tool = self.__callable.get(tool_name)
            streaming_config = tool.streaming_config if tool else None

            visible_params = (
                apply_transform(tool_params, streaming_config.params_transform)
                if streaming_config
                else tool_params
            )
            param_defaults = tool.get_parameter_defaults() if tool else None

            new_span = (
                self.__span.new(
                    name=tool_name,
                    attributes={
                        "metadata": {"tool_call_id": tool_call_id},
                        "input": tool_params or {},
                    },
                )
                if self.__span
                else None
            )

            tool_metas.append(
                {
                    "tool_call_request": tool_call_request,
                    "tool_name": tool_name,
                    "tool_params": tool_params,
                    "streaming_config": streaming_config,
                    "visible_params": visible_params,
                    "param_defaults": param_defaults,
                    "span": new_span,
                    "verdict": verdict,
                    "asked": tool_call_id in asked,
                    "display": tool.display if tool else None,
                }
            )

            if streaming_config is not None and verdict.action == "run":
                yield ContentPartTool(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    params=visible_params,
                    display_text=format_display_text(
                        streaming_config.running_text,
                        tool_params,
                        defaults=param_defaults,
                    ),
                    status="running",
                    display=tool.display if tool else None,
                )

        results: Dict[int, Dict[str, Any]] = {}

        async for event in self.__execute(tool_metas, results):
            yield event

        tool_outputs: List = []
        for index, meta in enumerate(tool_metas):
            result = results[index]
            if isinstance(result.get("exec_response"), ChatCompletion):
                yield ToolCallResult(
                    chat_completion=result["exec_response"],
                    should_return_to_user=True,
                )
                return
            tool_outputs.append(
                {
                    "id": result["id"],
                    "response": result["response"],
                    "is_error": result["error"] is not None,
                }
            )

        yield ToolCallResult(
            chat_completion=ChatCompletion(
                sender=ChatCompletionSender.TOOL,
                content="",
                tool_call_responses=[
                    ToolCallResponse(
                        id=tool_output["id"],
                        response=tool_output["response"],
                        is_error=tool_output["is_error"],
                    )
                    for tool_output in tool_outputs
                ],
            ),
        )

    async def __verdict(
        self, tool_call_request: ToolCallRequest, earlier: List[ToolCallRequest]
    ) -> ToolCallVerdict:
        func = tool_call_request.function_call_request
        tool = self.__callable.get(func.name)
        if tool is not None and self.__check_call is not None:
            decision = await self.__check_call(tool_call_request, earlier)
            if decision is not None and decision.action == "decline":
                return ToolCallVerdict(
                    action="decline",
                    message=decision.message,
                    status="declined_by_rule",
                    rule_reason=decision.reason or None,
                )
            if decision is not None:
                self.__rule_asks.add(tool_call_request.id)
                return ToolCallVerdict(action="ask", params=func.params)
        # A tool the client runs reaches it only as a card, asked or not.
        asks = self.__asks_user(tool, func.name) or (
            tool is not None and tool.runs_on_client
        )
        return ToolCallVerdict(action="ask" if asks else "run", params=func.params)

    def __asks_user(self, tool: Optional[Tool], name: str) -> bool:
        return (
            tool is not None
            and tool.options.confirmation in ("ask", "require")
            and not (
                self.__allows_session(tool)
                and self.__session_approvals is not None
                and self.__session_approvals.allows(name)
            )
        )

    @staticmethod
    def __allows_session(tool: Optional[Tool]) -> bool:
        # A required ask holds for every call: the user cannot waive it for
        # the rest of the chat.
        return tool is not None and tool.options.confirmation != "require"

    async def __settle_approvals(
        self, tool_call_requests: List[ToolCallRequest], verdicts: List[ToolCallVerdict]
    ) -> AsyncIterator[ContentPartTool]:
        """Replaces each "ask" verdict with the user's decision, or with what
        the agent's approval route does when nobody can be asked. The root agent
        yields its cards while it waits; a sub-agent publishes them."""
        asked = [
            index for index, verdict in enumerate(verdicts) if verdict.action == "ask"
        ]
        approvals = self.__approvals or ToolApprovals()
        if approvals.mode == "run":
            for index in asked:
                verdicts[index] = ToolCallVerdict(
                    action="run",
                    params=tool_call_requests[index].function_call_request.params,
                )
            return
        cards = [self.__confirmation_event(tool_call_requests[i]) for i in asked]
        inbox = approvals.inbox
        if (
            approvals.mode == "decline"
            or inbox is None
            or not (approvals.root or inbox.publish(cards))
        ):
            for index in asked:
                verdicts[index] = ToolCallVerdict(
                    action="decline",
                    message=TOOL_CONFIRMATION_UNAVAILABLE_MESSAGE,
                    status="confirmation_unavailable",
                )
            return
        if approvals.root:
            for card in cards:
                yield card
        outcome = await inbox.await_decisions(
            [tool_call_requests[index].id for index in asked]
        )
        for index in asked:
            verdicts[index] = await self.__rechecked(
                tool_call_requests[index],
                tool_call_requests[:index],
                self.__decided(tool_call_requests[index], outcome),
            )
        if not approvals.root:
            inbox.publish(
                [
                    self.__resolution_event(tool_call_requests[index], verdicts[index])
                    for index in asked
                ]
            )

    def __decided(
        self, tool_call_request: ToolCallRequest, outcome: ApprovalOutcome
    ) -> ToolCallVerdict:
        func = tool_call_request.function_call_request
        decision = outcome.decisions.get(tool_call_request.id)
        if decision is None:
            return ToolCallVerdict(
                action="decline",
                message=(
                    TOOL_NOT_CONFIRMED_MESSAGE
                    if outcome.superseded
                    else TOOL_APPROVAL_TIMED_OUT_MESSAGE
                ),
                status="confirmation_denied",
            )
        if decision.decision == "deny":
            return ToolCallVerdict(
                action="decline",
                message=TOOL_DECLINED_MESSAGE,
                status="confirmation_denied",
            )
        if (
            decision.decision == "allow_session"
            and tool_call_request.id not in self.__rule_asks
            and self.__session_approvals is not None
            and self.__allows_session(self.__callable.get(func.name))
        ):
            self.__session_approvals.allow(func.name)
        if decision.params is None or decision.params == func.params:
            return ToolCallVerdict(action="run", params=func.params)
        tool = self.__callable.get(func.name)
        if tool is not None and not tool.reports_edited_inputs:
            return ToolCallVerdict(action="run", params=decision.params)
        return ToolCallVerdict(
            action="run",
            params=decision.params,
            note=(
                "The user edited the inputs before running. Inputs used: "
                f"{json.dumps(decision.params, ensure_ascii=False)}"
            ),
        )

    async def __rechecked(
        self,
        tool_call_request: ToolCallRequest,
        earlier: List[ToolCallRequest],
        verdict: ToolCallVerdict,
    ) -> ToolCallVerdict:
        """The rules saw the model's inputs; inputs the user edited are a new
        call they must see too. The user approved these inputs, so a rule's
        ask is met and only its decline stands."""
        func = tool_call_request.function_call_request
        if (
            verdict.action != "run"
            or verdict.params == func.params
            or self.__check_call is None
        ):
            return verdict
        edited = ToolCallRequest(
            id=tool_call_request.id,
            function_call_request=FunctionCallRequest(
                name=func.name, params=verdict.params
            ),
        )
        decision = await self.__check_call(edited, earlier)
        if decision is not None and decision.action == "decline":
            return ToolCallVerdict(
                action="decline",
                message=decision.message,
                status="declined_by_rule",
                rule_reason=decision.reason or None,
            )
        return verdict

    def __resolution_event(
        self, tool_call_request: ToolCallRequest, verdict: ToolCallVerdict
    ) -> ContentPartTool:
        """What replaces a sub-agent's card in the root agent's output once
        decided; the sub-agent's own events show the call inside its task."""
        declined = verdict.action == "decline"
        name = tool_call_request.function_call_request.name
        tool: Optional[Tool] = self.__callable.get(name)
        return ContentPartTool(
            tool_call_id=tool_call_request.id,
            name=name,
            params=verdict.params,
            response={"status": verdict.status} if declined else None,
            display_text=CONFIRMATION_DECLINED_DISPLAY_TEXT if declined else None,
            status="completed" if declined else "running",
            display=tool.display if tool else None,
            rule_reason=verdict.rule_reason,
        )

    def __confirmation_event(
        self, tool_call_request: ToolCallRequest
    ) -> ContentPartTool:
        func = tool_call_request.function_call_request
        tool: Optional[Tool] = self.__callable.get(func.name)
        display = tool.display if tool else None
        return ContentPartTool(
            tool_call_id=tool_call_request.id,
            name=func.name,
            params=func.params,
            response={
                "status": "confirmation_required",
                "app": display.app if display else None,
                "icon_url": display.icon_url if display else None,
                "title": display.title if display else None,
                "parameters_spec": tool.get_parameters_spec() if tool else None,
                "allow_session": tool_call_request.id not in self.__rule_asks
                and self.__allows_session(tool),
                "asks": tool_call_request.id in self.__rule_asks
                or self.__asks_user(tool, func.name),
            },
            display_text=CONFIRMATION_REQUIRED_DISPLAY_TEXT,
            status="completed",
            display=display,
        )

    async def __execute_one(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        tool_call_request = meta["tool_call_request"]
        tool_name = meta["tool_name"]
        tool_params = meta["tool_params"]
        tool_span = meta["span"]
        verdict: ToolCallVerdict = meta["verdict"]
        if verdict.action == "decline":
            if tool_span:
                tool_span.end(attributes={"output": verdict.message})
            return {
                "id": tool_call_request.id,
                "response": verdict.message,
                "exec_response": None,
                "error": None,
            }
        try:
            if (
                self.__tool_choice
                and self.__tool_choice not in ("auto", "none", "required")
                and tool_name != self.__tool_choice
            ):
                raise Exception(
                    f"Error: tool_choice set to {self.__tool_choice} "
                    f"but {tool_name} was called. "
                    f"{self.__tool_choice} must be used."
                )

            if tool_name not in self.__callable:
                raise ValueError(
                    f"Tool {tool_name} not found. Please provide a valid tool name."
                )
            exec_response = await self.__tools_registry.execute(
                name=tool_name, params=tool_params
            )
            blocked_message: Optional[str] = None
            replaced = False
            rule_reason: Optional[str] = None
            if self.__check_result is not None:
                decision = await self.__check_result(tool_call_request, exec_response)
                if decision is not None and decision.action == "block":
                    exec_response, blocked_message = None, decision.message
                elif decision is not None:
                    exec_response, replaced = decision.value, True
                if decision is not None and decision.reason:
                    rule_reason = decision.reason
            sub_agent_parts: List[ContentPart] = []
            if blocked_message is not None:
                tool_response = blocked_message
            elif isinstance(exec_response, SubAgentToolResult):
                sub_agent_parts = exec_response.content_parts
                tool_response = exec_response.text
                exec_response = {"text": exec_response.text}
            else:
                streaming_config = meta["streaming_config"]
                llm_transform = (
                    streaming_config.llm_response_transform
                    if streaming_config is not None
                    else None
                )
                if llm_transform is not None:
                    tool_response = _tool_text(
                        apply_transform(
                            _parse_result_dict(exec_response), llm_transform
                        )
                    )
                else:
                    tool_response = _tool_text(exec_response)
            if verdict.note:
                tool_response = f"{verdict.note}\n\n{tool_response}"
            if tool_span:
                tool_span.end(attributes={"output": tool_response})
            return {
                "id": tool_call_request.id,
                "response": tool_response,
                "exec_response": exec_response,
                "sub_agent_parts": sub_agent_parts,
                "blocked": blocked_message is not None,
                "replaced": replaced,
                "rule_reason": rule_reason,
                "error": None,
            }
        except ValueError as value_error:
            error = str(value_error)
            if tool_span:
                tool_span.end(attributes={"output": error})
            return {
                "id": tool_call_request.id,
                "response": error,
                "exec_response": None,
                "error": "value_error",
            }
        except Exception as exception:
            error = str(exception)
            if self.__log_fn:
                self.__log_fn({"Error": error})
            if tool_span:
                tool_span.end(attributes={"output": error})
            return {
                "id": tool_call_request.id,
                "response": error,
                "exec_response": None,
                "error": "exception",
            }

    async def __execute(
        self, tool_metas: List[Dict[str, Any]], results: Dict[int, Dict[str, Any]]
    ) -> AsyncIterator[ContentPartTool]:
        """Runs the calls, concurrently or one after the other, yielding each
        call's events as it finishes. The root agent also yields the approval
        cards its sub-agents publish meanwhile."""
        events: "asyncio.Queue[Union[ContentPartTool, Tuple[int, Dict[str, Any]]]]" = (
            asyncio.Queue()
        )

        async def execute_and_enqueue(index: int) -> None:
            await events.put((index, await self.__execute_one(tool_metas[index])))

        started = len(tool_metas) if self.__concurrent_tool_execution else 1
        tasks = [
            asyncio.create_task(execute_and_enqueue(index)) for index in range(started)
        ]
        with self.__showing_sub_agent_cards(events):
            try:
                while len(results) < len(tool_metas):
                    event = await events.get()
                    if isinstance(event, ContentPartTool):
                        yield event
                        continue
                    index, result = event
                    results[index] = result
                    completion = self.__completion_event(tool_metas[index], result)
                    if completion is not None:
                        yield completion
                    for child_event in self.__child_tool_events(
                        tool_metas[index], result
                    ):
                        yield child_event
                    if started < len(tool_metas):
                        tasks.append(asyncio.create_task(execute_and_enqueue(started)))
                        started += 1
            except (asyncio.CancelledError, GeneratorExit):
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise

    def __showing_sub_agent_cards(
        self, events: "asyncio.Queue[Any]"
    ) -> ContextManager[None]:
        approvals = self.__approvals
        if approvals is None or not approvals.root or approvals.inbox is None:
            return nullcontext()
        return approvals.inbox.showing_cards_in(events)

    def __child_tool_events(
        self, meta: Dict[str, Any], result: Dict[str, Any]
    ) -> List[ContentPartTool]:
        parent_tool_call_id = meta["tool_call_request"].id
        return [
            self.__copy_child_tool_event(parent_tool_call_id, child_part.tool)
            for child_part in result.get("sub_agent_parts") or []
            if child_part.type == "tool" and child_part.tool is not None
        ]

    def __copy_child_tool_event(
        self, parent_tool_call_id: str, child_tool: ContentPartTool
    ) -> ContentPartTool:
        return child_tool.model_copy(
            update={"tool_call_id": f"{parent_tool_call_id}:{child_tool.tool_call_id}"}
        )

    def __completion_event(
        self, meta: Dict[str, Any], result: Dict[str, Any]
    ) -> Optional[ContentPartTool]:
        tool_call_request = meta["tool_call_request"]
        streaming_config = meta["streaming_config"]
        verdict: ToolCallVerdict = meta["verdict"]

        # Always streamed: the UI resolves the confirmation card by this id.
        if verdict.action == "decline":
            return ContentPartTool(
                tool_call_id=tool_call_request.id,
                name=meta["tool_name"],
                params=meta["visible_params"],
                response={"status": verdict.status},
                display_text=CONFIRMATION_DECLINED_DISPLAY_TEXT,
                status="completed",
                display=meta["display"],
                rule_reason=verdict.rule_reason,
            )

        if result.get("blocked"):
            return ContentPartTool(
                tool_call_id=tool_call_request.id,
                name=meta["tool_name"],
                params=meta["visible_params"],
                response={"status": "blocked_by_rule"},
                display_text=RESULT_BLOCKED_DISPLAY_TEXT,
                status="completed",
                display=meta["display"],
                rule_reason=result["rule_reason"],
            )

        replaced = bool(result.get("replaced"))
        # A card the user answered shows the call's outcome in its place, and
        # a result a rule replaced shows that a rule changed it.
        if streaming_config is None and (meta["asked"] or replaced):
            return ContentPartTool(
                tool_call_id=tool_call_request.id,
                name=meta["tool_name"],
                params=meta["visible_params"],
                response=_parse_result_dict(result["exec_response"]),
                status="error" if result["error"] else "completed",
                display=meta["display"],
                replaced_by_rule=replaced,
                rule_reason=result.get("rule_reason"),
            )

        if result["error"]:
            if streaming_config is None:
                return None
            return ContentPartTool(
                tool_call_id=tool_call_request.id,
                name=meta["tool_name"],
                params=meta["visible_params"],
                display_text=None,
                status="error",
                display=meta["display"],
            )

        if streaming_config is None:
            return None

        completed_text = (
            streaming_config.completed_text or streaming_config.running_text
        )
        result_dict = _parse_result_dict(result["exec_response"])
        visible_response = apply_transform(
            result_dict, streaming_config.response_transform
        )
        return ContentPartTool(
            tool_call_id=tool_call_request.id,
            name=meta["tool_name"],
            params=meta["visible_params"],
            response=visible_response,
            display_text=format_display_text(
                completed_text,
                meta["tool_params"],
                result_dict,
                defaults=meta["param_defaults"],
            ),
            status="completed",
            display=meta["display"],
            replaced_by_rule=replaced,
            rule_reason=result.get("rule_reason"),
        )
