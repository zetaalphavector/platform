from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.agent_state.attempt_lineage import (
    Attempt,
    AttemptLineage,
    Score,
    best_accepted,
)
from zav.agents_sdk.adapters.tools.sources.self_improvement.attempts.scorer import (
    ScoreContext,
    Scorer,
    ScorerGroup,
)
from zav.agents_sdk.adapters.tools.tools_source import (
    ToolsSource,
    ToolsSourceConfiguration,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, ToolStreamingConfig

# The transcript recorder finds committed attempt ids by this exact line.
ATTEMPT_ID_LINE = "attempt_id: {attempt_id}"

_DEFAULT_INSTRUCTIONS = """\
## Attempts

This question is a known task: your candidate answers are scored by an external \
scorer that you cannot see or influence. Do not answer directly. Work in attempts:

1. State a hypothesis: what you will change and why it should score better.
2. Produce the candidate artifact in full.
3. Call `commit_attempt` with both. The scorer runs and the verdict comes back.

An attempt is accepted only if it passes the scorer and matches or beats the \
best accepted score so far. Rejected attempts stay in the lineage with their \
score so you can learn from them; read any earlier candidate with `read_attempt`. \
Never resubmit a rejected artifact unchanged. When you stop, your answer to the \
user is the best accepted artifact.\
"""

_COMMIT_TOOL_DESCRIPTION = """\
Score a candidate artifact and record it in the attempt lineage. Returns the \
score, the verdict (accepted or rejected against the best accepted attempt so \
far), and the attempt id. Commit whole artifacts, not diffs.\
"""

_READ_TOOL_DESCRIPTION = """\
Read an earlier attempt: its hypothesis, score, verdict, notes and full artifact.\
"""


class AttemptsToolsSourceConfiguration(ToolsSourceConfiguration):
    enabled: bool = Field(
        False,
        description=(
            "Offer `commit_attempt` / `read_attempt` and the attempt instructions "
            "on turns the rubric context source recognised as a known task; other "
            "turns are answered normally."
        ),
    )
    scorer: Optional[str] = Field(
        None,
        description="source_name of the scorer to gate with. Default: first enabled.",
    )
    min_improvement: float = Field(
        0.0,
        description=(
            "How much a new attempt must beat the best accepted score by. 0 accepts "
            "ties, which is the right default for deterministic scorers; raise it to "
            "the judge's observed spread for noisy scorers."
        ),
    )
    digest_limit: int = Field(
        20, description="Most recent attempts listed in the system prompt each turn."
    )
    max_attempts: Optional[int] = Field(
        None,
        description=(
            "Commits allowed for this task, counting rejected ones. With a pass/fail "
            "scorer and a small answer space the gate is otherwise a search oracle; "
            "a budget makes each attempt cost a hypothesis. None means unlimited."
        ),
    )
    instructions: Optional[str] = Field(
        None, description="Override the default attempts instructions."
    )


class AttemptsToolsSource(ToolsSource):
    """The fitness gate over the shared ``AttemptLineage``, as a tools source:
    `commit_attempt` and `read_attempt` plus the "## Attempts" instructions. It
    engages on a turn only when the rubric context source selected a rubric for
    the user's message while the conversation was processed; on other turns it
    contributes no tools and no prompt."""

    source_name = "attempts_tools"
    source_description = "Record, score and compare the agent's attempts at a task."
    source_gate = "attempt_lineage"

    def __init__(
        self,
        lineage: AttemptLineage,
        scorers: List[Scorer],
        enabled: bool,
        scorer_name: Optional[str],
        min_improvement: float,
        digest_limit: int,
        instructions: Optional[str],
        max_attempts: Optional[int],
        defer_tools: Optional[bool] = None,
    ) -> None:
        self.enabled = enabled
        self.defer_tools = defer_tools
        self.__lineage = lineage
        self.__scorers = scorers
        self.__enabled = enabled
        self.__scorer_name = scorer_name
        self.__min_improvement = min_improvement
        self.__digest_limit = digest_limit
        self.__instructions = instructions or _DEFAULT_INSTRUCTIONS
        self.__max_attempts = max_attempts

    async def get_tools(self) -> List[Tool]:
        if self.__store() is None:
            return []
        return [self.__commit_tool(), self.__read_tool()]

    async def to_prompt(self) -> str:
        store = self.__store()
        if store is None:
            return ""
        attempts = await store.list_attempts()
        return "\n\n".join([self.__instructions, self.__digest(attempts)])

    async def commit(
        self,
        hypothesis: str,
        artifact: str,
        notes: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        store = self.__store()
        if store is None or self.__lineage.rubric is None:
            return "Error: this turn is not a scored task."
        scorer = self.__scorer()
        if scorer is None:
            return "Error: no scorer is configured; nothing can be committed."
        if not artifact.strip():
            return "Error: the artifact is empty."

        attempts = await store.list_attempts()
        if self.__max_attempts is not None and len(attempts) >= self.__max_attempts:
            return (
                f"Error: the attempt budget of {self.__max_attempts} is spent; no more "
                "commits are accepted. Answer with the best accepted artifact, or the "
                "best candidate you have if none was accepted."
            )
        best = best_accepted(attempts)
        score = await scorer.score(
            artifact,
            ScoreContext(
                rubric=self.__lineage.rubric, hypothesis=hypothesis, notes=notes
            ),
        )
        accepted = score.passed and self.__beats(score, best)
        attempt = await store.add(
            Attempt(
                parent_id=parent_id or (best.id if best else None),
                hypothesis=hypothesis,
                artifact=artifact,
                notes=notes,
                score=score,
                verdict="accepted" if accepted else "rejected",
            )
        )
        return self.__render_verdict(attempt, score, best, len(attempts))

    async def read(self, attempt_id: str) -> str:
        store = self.__store()
        if store is None:
            return "Error: this turn is not a scored task."
        attempt = await store.get(attempt_id)
        if attempt is None:
            return f"Error: no attempt with id {attempt_id!r}."
        lines = [
            ATTEMPT_ID_LINE.format(attempt_id=attempt.id),
            f"parent: {attempt.parent_id or 'none'}",
            f"verdict: {attempt.verdict}",
            f"score: {attempt.score.render() if attempt.score else 'unscored'}",
            f"hypothesis: {attempt.hypothesis}",
        ]
        if attempt.notes:
            lines.append(f"notes: {attempt.notes}")
        if attempt.score and attempt.score.details:
            lines.append(f"scorer details: {attempt.score.details}")
        lines.extend(["", "artifact:", attempt.artifact])
        return "\n".join(lines)

    def __beats(self, score: Score, best: Optional[Attempt]) -> bool:
        if best is None or best.score is None:
            return True
        return score.value >= best.score.value + self.__min_improvement

    def __render_verdict(
        self,
        attempt: Attempt,
        score: Score,
        previous_best: Optional[Attempt],
        committed_before: int,
    ) -> str:
        lines = [
            ATTEMPT_ID_LINE.format(attempt_id=attempt.id),
            f"verdict: {attempt.verdict}",
            f"score: {score.render()}",
        ]
        if previous_best and previous_best.score:
            lines.append(
                f"best before this attempt: {previous_best.id} at "
                f"{previous_best.score.render()}"
            )
        else:
            lines.append("best before this attempt: none")
        if score.details:
            lines.append(f"scorer details: {score.details}")
        if attempt.verdict == "rejected":
            if not score.passed:
                lines.append("rejected because the scorer did not pass it.")
            else:
                lines.append(
                    "rejected because it does not improve on the best accepted score."
                )
        remaining = self.__remaining_attempts(committed_before + 1)
        if remaining is not None:
            lines.append(f"attempts remaining: {remaining}")
        return "\n".join(lines)

    def __remaining_attempts(self, committed: int) -> Optional[int]:
        if self.__max_attempts is None:
            return None
        return max(0, self.__max_attempts - committed)

    def __digest(self, attempts: List[Attempt]) -> str:
        if not attempts:
            budget = (
                f" Attempts available: {self.__max_attempts}."
                if self.__max_attempts is not None
                else ""
            )
            task = self.__lineage.task if self.__lineage.task else "task"
            return f"### Lineage of {task}\n\nNo attempts committed yet.{budget}"
        best = best_accepted(attempts)
        task = self.__lineage.task if self.__lineage.task else "task"
        header = (
            f"### Lineage of {task} (best: {best.id} at {best.score.render()})"  # type: ignore  # noqa: E501
            if best and best.score
            else f"### Lineage of {task} (no accepted attempt yet)"
        )
        recent = attempts[-self.__digest_limit :]
        lines = [header, ""]
        remaining = self.__remaining_attempts(len(attempts))
        if remaining is not None:
            lines.append(f"Attempts remaining: {remaining} of {self.__max_attempts}.")
        if len(attempts) > len(recent):
            lines.append(f"({len(attempts) - len(recent)} earlier attempts omitted)")
        for attempt in recent:
            score = attempt.score.render() if attempt.score else "unscored"
            parent = f", parent {attempt.parent_id}" if attempt.parent_id else ""
            lines.append(
                f"- {attempt.id} ({attempt.verdict}, {score}{parent}): "
                f"{attempt.hypothesis}"
            )
        return "\n".join(lines)

    def __commit_tool(self) -> Tool:
        async def commit_attempt(
            hypothesis: str,
            artifact: str,
            notes: Optional[str] = None,
            parent_id: Optional[str] = None,
        ) -> str:
            return await self.commit(hypothesis, artifact, notes, parent_id)

        return Tool(
            name="commit_attempt",
            description=_COMMIT_TOOL_DESCRIPTION,
            executable=commit_attempt,
            parameters_spec={
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": (
                            "What this attempt changes and why it should score "
                            "better."
                        ),
                    },
                    "artifact": {
                        "type": "string",
                        "description": "The complete candidate answer or artifact.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional observations worth keeping.",
                    },
                    "parent_id": {
                        "type": "string",
                        "description": (
                            "Attempt this one builds on. Defaults to the best "
                            "accepted attempt."
                        ),
                    },
                },
                "required": ["hypothesis", "artifact"],
            },
            streaming_config=ToolStreamingConfig(
                running_text="Scoring attempt...",
                completed_text="Scored attempt",
            ),
        )

    def __read_tool(self) -> Tool:
        async def read_attempt(attempt_id: str) -> str:
            return await self.read(attempt_id)

        return Tool(
            name="read_attempt",
            description=_READ_TOOL_DESCRIPTION,
            executable=read_attempt,
            parameters_spec={
                "type": "object",
                "properties": {
                    "attempt_id": {"type": "string", "description": "Attempt id."}
                },
                "required": ["attempt_id"],
            },
        )

    def __store(self) -> Optional[AttemptLineage]:
        if not self.__enabled or not self.__lineage.active:
            return None
        return self.__lineage

    def __scorer(self) -> Optional[Scorer]:
        if not self.__enabled:
            return None
        active = [
            s
            for s in self.__scorers
            if is_source_active(s.source_name, s.enabled, None, None)
        ]
        if self.__scorer_name is None:
            return active[0] if active else None
        for scorer in active:
            if scorer.source_name == self.__scorer_name:
                return scorer
        logger.warning(f"Configured scorer '{self.__scorer_name}' is not enabled.")
        return None


class AttemptsToolsSourceFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        attempt_lineage: AttemptLineage,
        scorer_group: ScorerGroup = ScorerGroup(items=[]),
        attempts_tools_source_configuration: AttemptsToolsSourceConfiguration = (
            AttemptsToolsSourceConfiguration()
        ),
    ) -> AttemptsToolsSource:
        configuration = attempts_tools_source_configuration
        return AttemptsToolsSource(
            lineage=attempt_lineage,
            scorers=scorer_group.items,
            enabled=configuration.enabled,
            defer_tools=configuration.defer_tools,
            scorer_name=configuration.scorer,
            min_improvement=configuration.min_improvement,
            digest_limit=configuration.digest_limit,
            instructions=configuration.instructions,
            max_attempts=configuration.max_attempts,
        )
