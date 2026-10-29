import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_delegation.agent_delegation_provider import (
    AgentDelegationProvider,
)
from zav.agents_sdk.adapters.instructions.instructions_provider import (
    InstructionsProvider,
)
from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion
from zav.agents_sdk.adapters.llm_models.transcript_rendering import (
    render_transcript,
)
from zav.agents_sdk.adapters.memory.memory_provider import MemoryProvider
from zav.agents_sdk.adapters.skills.skills_provider import SkillsProvider
from zav.agents_sdk.adapters.tools.sources.self_improvement.refine.refinement_edit import (  # noqa: E501
    EditKind,
    RefinementEdit,
    apply_edit,
    available_kinds,
    parse_edits,
)
from zav.agents_sdk.adapters.tools.sources.self_improvement.refine.refinement_ledger import (  # noqa: E501
    RefinementLedger,
)
from zav.agents_sdk.domain.agent_creator import AgentCreator, run_sub_agent
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_DEFAULT_INSTRUCTIONS = """\
You refine the harness of another agent. Below are the agent's current harness \
and the evidence: an observation it made, the transcript of its turn so far, or \
both.

Propose the smallest set of evidence-backed edits that will still hold in \
future turns. Allowed kinds:
{kinds}

Rules: do not duplicate what the harness already holds; do not store the task's \
answer, scores or reference solutions; prefer no edit over a speculative one. \
Reply with only a JSON array, possibly empty, of objects with the fields \
`kind`, `name` (skills, prompt notes and sub-agents, lowercase-hyphenated), \
`description` (skills and sub-agents, one sentence), `content`, `tools` \
(sub-agents: tool source names from the harness), `trigger` (the evidence, one \
sentence) and `intended_effect` (one sentence).\
"""

_KIND_DESCRIPTIONS = {
    "memory": "- `memory`: a durable fact about the tools, environment or user.",
    "skill": "- `skill`: a repeatable procedure worth a named recipe.",
    "prompt_note": "- `prompt_note`: a narrow behavioural rule.",
    "sub_agent": (
        "- `sub_agent`: a specialist the agent can delegate a task to; `content` "
        "is its system prompt and `tools` the tool sources it may use."
    ),
}


class RefinerConfiguration(BaseModel):
    refiner_agent_identifier: Optional[str] = Field(
        None,
        description=(
            "Agent that proposes the edits. Defaults to a fresh copy of the "
            "current agent; prefer a lean agent with tools and memory off."
        ),
    )
    max_edits: int = Field(
        3, description="Edits kept from one refinement; the rest are dropped."
    )
    max_transcript_chars: int = Field(
        24_000, description="Tail of the rendered transcript handed to the refiner."
    )
    ledger_path: Optional[str] = Field(
        "refinements.jsonl",
        description=(
            "JSON-lines file recording every edit with its trigger, intended "
            "effect and result. None disables the ledger."
        ),
    )
    instructions: Optional[str] = Field(
        None,
        description=(
            "Override the refiner instructions; `{kinds}` is replaced by the "
            "allowed edit kinds."
        ),
    )


class Refiner:
    """One refinement, Prime's `/refine` planning and apply: a lean agent reads
    the current harness and the evidence, proposes typed edits, and they are
    written to the memory, skills, instructions and delegation providers'
    writable targets.
    Only kinds with a writable target are offered. Shared by the
    `request_refinement` tool (the agent asks) and the `refine` interjection
    source (the harness decides)."""

    def __init__(
        self,
        agent_creator: AgentCreator,
        memory_provider: MemoryProvider,
        skills_provider: SkillsProvider,
        instructions_provider: InstructionsProvider,
        agent_delegation_provider: AgentDelegationProvider,
        refiner_agent_identifier: Optional[str],
        max_edits: int,
        max_transcript_chars: int,
        ledger: Optional[RefinementLedger],
        instructions: Optional[str],
    ) -> None:
        self.__agent_creator = agent_creator
        self.__memory_provider = memory_provider
        self.__skills_provider = skills_provider
        self.__instructions_provider = instructions_provider
        self.__agent_delegation_provider = agent_delegation_provider
        self.__refiner_agent_identifier = refiner_agent_identifier
        self.__max_edits = max_edits
        self.__max_transcript_chars = max_transcript_chars
        self.__ledger = ledger
        self.__instructions = instructions or _DEFAULT_INSTRUCTIONS
        self.__kinds: List[EditKind] = available_kinds(
            memory_provider,
            skills_provider,
            instructions_provider,
            agent_delegation_provider,
        )

    @property
    def kinds(self) -> List[EditKind]:
        return self.__kinds

    async def refine(
        self,
        focus: str,
        transcript: Optional[List[ChatCompletion]],
        mode: str,
    ) -> str:
        """Runs one refinement and returns what was kept, in one sentence for the
        agent. Never raises."""
        if not self.__kinds:
            return "Refinement is not possible: nothing is writable."
        try:
            reply = await run_sub_agent(
                agent_creator=self.__agent_creator,
                prompt=await self.__prompt(focus, transcript),
                target_identifier=self.__refiner_agent_identifier,
            )
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return "Refinement failed."
        edits = [
            edit
            for edit in parse_edits(reply.content or "")
            if edit.kind in self.__kinds
        ][: self.__max_edits]
        kept: List[str] = []
        for edit in edits:
            try:
                result = await apply_edit(
                    edit,
                    self.__memory_provider,
                    self.__skills_provider,
                    self.__instructions_provider,
                    self.__agent_delegation_provider,
                )
                kept.append(result)
            except Exception as e:
                result = f"error: {e}"
                logger.error(f"Refinement edit {edit.kind} not applied: {e}")
            self.__record(edit, focus, mode, result)
        if not kept:
            return "Refinement proposed no edits."
        return (
            "Refinement kept: " + "; ".join(kept) + ". It is in your persistent "
            "state from the next turn on."
        )

    async def render_harness(self) -> str:
        lines: List[str] = []
        for store in self.__memory_provider.active_stores:
            for entry in await store.load_context():
                lines.append(f"memory: {entry.content}")
        for source in self.__skills_provider.active_sources:
            for name, skill in (await source.discover()).items():
                lines.append(f"skill {name}: {skill.description}")
        for instruction_source in self.__instructions_provider.active_sources:
            text = await instruction_source.to_prompt()
            if text:
                lines.append(f"instructions ({instruction_source.source_name}): {text}")
        for agent in await self.__agent_delegation_provider.list_agents():
            lines.append(f"sub-agent {agent.name}: {agent.description}")
        writable = self.__agent_delegation_provider.writable_source
        if writable is not None and writable.available_tools:
            names = ", ".join(writable.available_tools)
            lines.append(f"tool sources a sub-agent may use: {names}")
        return "\n".join(lines) or "(empty)"

    def render_transcript(self, transcript: List[ChatCompletion]) -> str:
        return render_transcript(transcript, self.__max_transcript_chars)

    async def __prompt(
        self, focus: str, transcript: Optional[List[ChatCompletion]]
    ) -> str:
        kinds = "\n".join(_KIND_DESCRIPTIONS[kind] for kind in self.__kinds)
        sections = [self.__instructions.replace("{kinds}", kinds)]
        if focus:
            sections.append(f"<focus>\n{focus}\n</focus>")
        sections.append(f"<harness>\n{await self.render_harness()}\n</harness>")
        if transcript:
            sections.append(
                f"<transcript>\n{self.render_transcript(transcript)}\n</transcript>"
            )
        return "\n\n".join(sections)

    def __record(
        self, edit: RefinementEdit, focus: str, mode: str, result: str
    ) -> None:
        if self.__ledger is None:
            return
        record = {
            "id": uuid.uuid4().hex[:8],
            "at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "focus": focus or None,
            "result": result,
            **json.loads(edit.json()),
        }
        try:
            self.__ledger.append(record)
        except Exception as e:
            logger.error(f"Failed to write refinement ledger: {e}")


class RefinerFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls,
        agent_creator: AgentCreator,
        memory_provider: MemoryProvider,
        skills_provider: SkillsProvider,
        instructions_provider: InstructionsProvider,
        agent_delegation_provider: AgentDelegationProvider,
        refiner_configuration: RefinerConfiguration = RefinerConfiguration(),
    ) -> Refiner:
        configuration = refiner_configuration
        return Refiner(
            agent_creator=agent_creator,
            memory_provider=memory_provider,
            skills_provider=skills_provider,
            instructions_provider=instructions_provider,
            agent_delegation_provider=agent_delegation_provider,
            refiner_agent_identifier=configuration.refiner_agent_identifier,
            max_edits=configuration.max_edits,
            max_transcript_chars=configuration.max_transcript_chars,
            ledger=(
                RefinementLedger(Path(configuration.ledger_path))
                if configuration.ledger_path
                else None
            ),
            instructions=configuration.instructions,
        )
