from typing import Callable, Dict, List, Optional, Union

from zav.pydantic_compat import PYDANTIC_V2, BaseModel, ConfigDict

from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion
from zav.agents_sdk.domain.chat_message import ChatMessage
from zav.agents_sdk.domain.tools import ToolsRegistry

STREAM_TOOL_EVENTS_DEPRECATION_MESSAGE = (
    "stream_tool_events is deprecated; use stream_tool_progress instead."
)


class ChatTurnRequest(BaseModel):
    """Normalized inputs and options for one chat completion turn."""

    messages: Optional[List[ChatMessage]] = None
    completions: Optional[List[ChatCompletion]] = None
    max_tokens: Optional[int] = None
    bot_setup_description: Optional[str] = None
    functions: Optional[List[Dict]] = None
    tools: Optional[Union[ToolsRegistry, List[Dict]]] = None
    tool_choice: Optional[str] = None
    execute_tools: bool = True
    stream_tool_calls: bool = False
    stream_tool_progress: bool = False
    stream_tool_events: Optional[bool] = None
    preserve_streamed_content_parts: bool = False
    log_fn: Optional[Callable] = None
    max_nesting_level: int = 20
    reasoning_effort: Optional[str] = None
    logprobs: Optional[bool] = None
    parallel_tool_calls: Optional[bool] = None
    concurrent_tool_execution: bool = False
    seed: Optional[int] = None
    verbosity: Optional[str] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:

        class Config:
            arbitrary_types_allowed = True

    @property
    def should_stream_tool_progress(self) -> bool:
        return self.stream_tool_progress or bool(self.stream_tool_events)
