import asyncio
import json
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from zav.llm_tracing import Span
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
    ToolCallRequest,
    ToolCallResponse,
)
from zav.agents_sdk.domain.agent_creator import SubAgentToolResult
from zav.agents_sdk.domain.chat_message import ContentPart, ContentPartTool
from zav.agents_sdk.domain.tools import (
    ToolsRegistry,
    apply_transform,
    format_display_text,
)


class ToolCallResult(BaseModel):
    """Result of executing a batch of model-requested tool calls."""

    chat_completion: Optional[ChatCompletion] = None
    should_return_to_user: bool = False


def _parse_result_dict(exec_response: Any) -> Optional[dict]:
    if isinstance(exec_response, dict):
        return exec_response
    if isinstance(exec_response, str):
        try:
            return json.loads(exec_response)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


class ToolCallRunner:
    """Executes model-requested tools and optional streaming tool events."""

    def __init__(
        self,
        tools_registry: ToolsRegistry,
        tool_choice: Optional[str] = None,
        log_fn: Optional[Callable] = None,
        span: Optional[Span] = None,
        concurrent_tool_execution: bool = False,
    ) -> None:
        self.__tools_registry = tools_registry
        self.__tool_choice = tool_choice
        self.__log_fn = log_fn
        self.__span = span
        self.__concurrent_tool_execution = concurrent_tool_execution

    async def execute(
        self, tool_call_requests: Optional[List[ToolCallRequest]] = None
    ) -> ToolCallResult:
        async for event in self.execute_streaming(tool_call_requests):
            if isinstance(event, ToolCallResult):
                return event
        return ToolCallResult()

    async def execute_streaming(  # noqa: C901
        self, tool_call_requests: Optional[List[ToolCallRequest]] = None
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
                available_tools = ", ".join(self.__tools_registry.tools_index.keys())
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

        tool_metas: List[Dict[str, Any]] = []
        for tool_call_request in tool_call_requests:
            tool_call_id = tool_call_request.id
            func = tool_call_request.function_call_request
            tool_name = func.name
            tool_params = func.params

            tool = self.__tools_registry.tools_index.get(tool_name)
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
                }
            )

            if streaming_config is not None:
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
                )

        results: Dict[int, Dict[str, Any]] = {}

        if self.__concurrent_tool_execution:
            async for event in self.__execute_concurrent(tool_metas, results):
                yield event
        else:
            async for event in self.__execute_sequential(tool_metas, results):
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
            tool_outputs.append({"id": result["id"], "response": result["response"]})

        yield ToolCallResult(
            chat_completion=ChatCompletion(
                sender=ChatCompletionSender.TOOL,
                content="",
                tool_call_responses=[
                    ToolCallResponse(
                        id=tool_output["id"],
                        response=tool_output["response"],
                    )
                    for tool_output in tool_outputs
                ],
            ),
        )

    async def __execute_one(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        tool_call_request = meta["tool_call_request"]
        tool_name = meta["tool_name"]
        tool_params = meta["tool_params"]
        tool_span = meta["span"]
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

            exec_response = await self.__tools_registry.execute(
                name=tool_name, params=tool_params
            )
            sub_agent_parts: List[ContentPart] = []
            if isinstance(exec_response, SubAgentToolResult):
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
                    tool_response = str(
                        apply_transform(
                            _parse_result_dict(exec_response), llm_transform
                        )
                    )
                else:
                    tool_response = str(exec_response)
            if tool_span:
                tool_span.end(attributes={"output": tool_response})
            return {
                "id": tool_call_request.id,
                "response": tool_response,
                "exec_response": exec_response,
                "sub_agent_parts": sub_agent_parts,
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

    async def __execute_concurrent(
        self, tool_metas: List[Dict[str, Any]], results: Dict[int, Dict[str, Any]]
    ) -> AsyncIterator[ContentPartTool]:
        done_queue: asyncio.Queue[Tuple[int, Dict[str, Any]]] = asyncio.Queue()

        async def execute_and_enqueue(index: int, meta: Dict[str, Any]) -> None:
            result = await self.__execute_one(meta)
            await done_queue.put((index, result))

        tasks = [
            asyncio.create_task(execute_and_enqueue(index, meta))
            for index, meta in enumerate(tool_metas)
        ]

        try:
            for _ in range(len(tasks)):
                index, result = await done_queue.get()
                results[index] = result
                event = self.__completion_event(tool_metas[index], result)
                if event is not None:
                    yield event
                for child_event in self.__child_tool_events(tool_metas[index], result):
                    yield child_event
        except (asyncio.CancelledError, GeneratorExit):
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    async def __execute_sequential(
        self, tool_metas: List[Dict[str, Any]], results: Dict[int, Dict[str, Any]]
    ) -> AsyncIterator[ContentPartTool]:
        for index, meta in enumerate(tool_metas):
            result = await self.__execute_one(meta)
            results[index] = result
            event = self.__completion_event(meta, result)
            if event is not None:
                yield event
            for child_event in self.__child_tool_events(meta, result):
                yield child_event

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

        if result["error"]:
            if streaming_config is None:
                return None
            return ContentPartTool(
                tool_call_id=tool_call_request.id,
                name=meta["tool_name"],
                params=meta["visible_params"],
                display_text=None,
                status="error",
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
        )
