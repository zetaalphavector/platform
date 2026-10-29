import json
import re
from typing import List, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.attempt_lineage import AttemptLineage
from zav.agents_sdk.adapters.interjections.interjection_source import (
    InterjectionSource,
)
from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
)
from zav.agents_sdk.adapters.llm_models.chat_turn_request import InnerLoopContext
from zav.agents_sdk.adapters.tools.sources.self_improvement.attempts import (
    ATTEMPT_ID_LINE,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_ATTEMPT_ID_PATTERN = re.compile(
    re.escape(ATTEMPT_ID_LINE.format(attempt_id="")) + r"([0-9a-f]{8})"
)


class AttemptTranscriptRecorderInterjectionSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "After a tool batch that committed an attempt, save the transcript up "
            "to that point next to the attempt."
        ),
    )


class AttemptTranscriptRecorderInterjectionSource(InterjectionSource):
    """Silent interjection source: it never speaks, it only uses the between-tool-
    batches seam to attach the agent's transcript to freshly committed attempts,
    so a reviewer or the agent can later read how an attempt came about."""

    source_name = "attempt_transcript_recorder"
    source_description = "Records each attempt's transcript; never speaks."

    def __init__(self, attempt_lineage: AttemptLineage, enabled: bool) -> None:
        self.enabled = enabled and attempt_lineage.enabled
        self.__lineage = attempt_lineage
        self.__scanned = 0
        self.__seen: Set[str] = set()

    async def after_model_call(self, context: InnerLoopContext) -> Optional[str]:
        if not self.__lineage.active:
            return None
        # Only the completions added since the last call are scanned, and each
        # attempt id is looked up once; nothing touches the store on the calls
        # in between.
        new_ids = committed_attempt_ids(context.transcript[self.__scanned :])
        self.__scanned = len(context.transcript)
        new_ids -= self.__seen
        self.__seen |= new_ids
        if not new_ids:
            return None
        pending = []
        for attempt_id in sorted(new_ids):
            attempt = await self.__lineage.get(attempt_id)
            if attempt is not None and not attempt.has_transcript:
                pending.append(attempt)
        if not pending:
            return None
        transcript = [json.loads(c.json()) for c in context.transcript]
        for attempt in pending:
            try:
                await self.__lineage.save_transcript(attempt.id, transcript)
            except Exception as e:
                logger.error(f"Failed to save transcript for attempt {attempt.id}: {e}")
        return None


def committed_attempt_ids(transcript: List[ChatCompletion]) -> Set[str]:
    ids: Set[str] = set()
    for completion in transcript:
        if completion.sender != ChatCompletionSender.TOOL:
            continue
        for response in completion.tool_call_responses or []:
            ids.update(_ATTEMPT_ID_PATTERN.findall(response.response or ""))
    return ids


class AttemptTranscriptRecorderInterjectionSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        attempt_lineage: AttemptLineage,
        attempt_transcript_recorder_interjection_source_configuration: (
            AttemptTranscriptRecorderInterjectionSourceConfiguration
        ) = AttemptTranscriptRecorderInterjectionSourceConfiguration(),
    ) -> AttemptTranscriptRecorderInterjectionSource:
        configuration = attempt_transcript_recorder_interjection_source_configuration
        return AttemptTranscriptRecorderInterjectionSource(
            attempt_lineage=attempt_lineage, enabled=configuration.enabled
        )
