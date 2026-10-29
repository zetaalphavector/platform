from typing import Dict, List, Optional, Tuple

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.interjections.interjection_source import (
    InterjectionSource,
)
from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
)
from zav.agents_sdk.adapters.llm_models.chat_turn_request import InnerLoopContext
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

# ToolsRegistry.execute wraps every tool exception in this prefix; ToolCallRunner
# then returns the exception text as the tool response.
_TOOL_ERROR_PREFIX = "Error in executing tool "

_DEFAULT_MESSAGE = (
    "The tool `{tool_name}` has failed {count} times in a row with:\n"
    "{last_error}\n"
    "Retrying the same call is unlikely to help. Re-read the error, change the "
    "arguments or the approach, or use a different tool."
)


class RepeatedToolFailureInterjectionSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable the repeated-tool-failure watch.")
    max_consecutive_failures: int = Field(
        3,
        description=(
            "Interject once the same tool has failed this many times in a row."
        ),
    )
    message: Optional[str] = Field(
        None,
        description=(
            "Override the interjection text. Supports {tool_name}, {count} and "
            "{last_error} placeholders."
        ),
    )


class RepeatedToolFailureInterjectionSource(InterjectionSource):
    """Speaks up when the agent keeps calling one tool and it keeps failing."""

    source_name = "repeated_tool_failure"
    source_description = "Speaks up when the same tool keeps failing."

    def __init__(
        self,
        enabled: bool,
        max_consecutive_failures: int,
        message: Optional[str],
    ) -> None:
        self.enabled = enabled
        self.__max_consecutive_failures = max(1, max_consecutive_failures)
        self.__message = message or _DEFAULT_MESSAGE
        self.__last_reported_run: Optional[Tuple[str, int]] = None

    async def after_model_call(self, context: InnerLoopContext) -> Optional[str]:
        streak = trailing_failure_streak(context.transcript)
        if streak is None:
            return None
        tool_name, count, last_error = streak
        # Interject once each time the streak crosses another multiple of the
        # threshold. A parallel tool batch can jump the count past the exact
        # multiple, so compare crossings, not equality.
        crossed = count // self.__max_consecutive_failures
        if crossed == 0 or self.__last_reported_run == (tool_name, crossed):
            return None
        self.__last_reported_run = (tool_name, crossed)
        return self.__message.format(
            tool_name=tool_name, count=count, last_error=last_error
        )


def trailing_failure_streak(
    transcript: List[ChatCompletion],
) -> Optional[Tuple[str, int, str]]:
    """Walk back over the newest tool calls while they are failures of one tool.

    Returns (tool_name, consecutive_failures, last_error) or None when the
    newest tool call succeeded or there are no tool calls yet."""
    calls = tool_calls_newest_first(transcript)
    if not calls:
        return None
    streak_tool: Optional[str] = None
    last_error = ""
    count = 0
    for tool_name, response in calls:
        failed = response.startswith(_TOOL_ERROR_PREFIX)
        if not failed:
            break
        if streak_tool is None:
            streak_tool = tool_name
            last_error = response
        if tool_name != streak_tool:
            break
        count += 1
    if streak_tool is None or count == 0:
        return None
    return streak_tool, count, last_error


def tool_calls_newest_first(
    transcript: List[ChatCompletion],
) -> List[Tuple[str, str]]:
    names_by_call_id: Dict[str, str] = {}
    calls: List[Tuple[str, str]] = []
    for completion in transcript:
        for request in completion.tool_call_requests or []:
            names_by_call_id[request.id] = request.function_call_request.name
        if completion.sender != ChatCompletionSender.TOOL:
            continue
        for response in completion.tool_call_responses or []:
            calls.append(
                (
                    names_by_call_id.get(response.id, response.id),
                    response.response or "",
                )
            )
    calls.reverse()
    return calls


class RepeatedToolFailureInterjectionSourceFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        repeated_tool_failure_interjection_source_configuration: (
            RepeatedToolFailureInterjectionSourceConfiguration
        ) = RepeatedToolFailureInterjectionSourceConfiguration(),
    ) -> RepeatedToolFailureInterjectionSource:
        configuration = repeated_tool_failure_interjection_source_configuration
        return RepeatedToolFailureInterjectionSource(
            enabled=configuration.enabled,
            max_consecutive_failures=configuration.max_consecutive_failures,
            message=configuration.message,
        )
