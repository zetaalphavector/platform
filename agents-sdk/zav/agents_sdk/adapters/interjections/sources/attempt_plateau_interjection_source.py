from typing import List, Optional, Tuple

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.attempt_lineage import (
    Attempt,
    AttemptLineage,
    best_accepted,
)
from zav.agents_sdk.adapters.interjections.interjection_source import (
    InterjectionSource,
)
from zav.agents_sdk.adapters.llm_models.chat_turn_request import InnerLoopContext
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_DEFAULT_MESSAGE = (
    "The last {count} attempts were all rejected ({scores}); the best accepted "
    "score is still {best}. The current line of changes has plateaued. Before "
    "committing again, read the rejected attempts' scorer details, name what they "
    "have in common, and try a different kind of change rather than a variation "
    "of the last one."
)


class AttemptPlateauInterjectionSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Speak up when attempts stop improving.")
    max_consecutive_rejections: int = Field(
        3, description="Interject once this many attempts in a row were rejected."
    )
    message: Optional[str] = Field(
        None,
        description=(
            "Override the interjection text. Supports {count}, {scores} and {best}."
        ),
    )


class AttemptPlateauInterjectionSource(InterjectionSource):
    """AVO's stall signal, deterministic: the score curve stopped moving."""

    source_name = "attempt_plateau"
    source_description = "Nudges the agent when its attempt scores stop improving."

    def __init__(
        self,
        attempt_lineage: AttemptLineage,
        enabled: bool,
        max_consecutive_rejections: int,
        message: Optional[str],
    ) -> None:
        self.enabled = enabled and attempt_lineage.enabled
        self.__lineage = attempt_lineage
        self.__threshold = max(1, max_consecutive_rejections)
        self.__message = message or _DEFAULT_MESSAGE
        self.__last_reported: Optional[Tuple[str, int]] = None

    async def after_model_call(self, context: InnerLoopContext) -> Optional[str]:
        if not self.__lineage.active:
            return None
        attempts = await self.__lineage.list_attempts()
        streak = trailing_rejections(attempts)
        # Fire once per crossed multiple of the threshold; several attempts can
        # land in one parallel tool batch and skip the exact multiple.
        crossed = len(streak) // self.__threshold
        marker = (streak[0].id, crossed) if streak else None
        if crossed == 0 or marker == self.__last_reported:
            return None
        self.__last_reported = marker
        best = best_accepted(attempts)
        return self.__message.format(
            count=len(streak),
            scores=", ".join(
                a.score.render() if a.score else "unscored" for a in streak
            ),
            best=best.score.render() if best and best.score else "none",
        )


def trailing_rejections(attempts: List[Attempt]) -> List[Attempt]:
    streak: List[Attempt] = []
    for attempt in reversed(attempts):
        if attempt.verdict != "rejected":
            break
        streak.append(attempt)
    streak.reverse()
    return streak


class AttemptPlateauInterjectionSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        attempt_lineage: AttemptLineage,
        attempt_plateau_interjection_source_configuration: (
            AttemptPlateauInterjectionSourceConfiguration
        ) = AttemptPlateauInterjectionSourceConfiguration(),
    ) -> AttemptPlateauInterjectionSource:
        configuration = attempt_plateau_interjection_source_configuration
        return AttemptPlateauInterjectionSource(
            attempt_lineage=attempt_lineage,
            enabled=configuration.enabled,
            max_consecutive_rejections=configuration.max_consecutive_rejections,
            message=configuration.message,
        )
