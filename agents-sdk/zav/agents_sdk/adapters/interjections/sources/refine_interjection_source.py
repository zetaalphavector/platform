import json
import time
from typing import Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.interjections.interjection_source import (
    InterjectionSource,
)
from zav.agents_sdk.adapters.llm_models.chat_turn_request import InnerLoopContext
from zav.agents_sdk.adapters.tools.sources.self_improvement.refine import Refiner
from zav.agents_sdk.domain.agent_creator import AgentCreator, run_sub_agent
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_DEFAULT_GATE_INSTRUCTIONS = """\
You are the automatic refinement gate of an agent harness. Below are the agent's \
current harness (memories, skills, instructions) and the transcript of its turn \
so far. Decide whether this checkpoint should refine the harness.

Approve when the transcript holds evidence that will help future turns: a \
repeated failure and its cause, a reusable tactic, a durable fact about the \
tools or the environment. Reject one-off noise, unsupported hypotheses and \
transient tool output. Never approve storing the task's answer.

Reply with only a JSON object: {"should_refine": true|false, "rationale": \
"one sentence", "instructions": "what to persist, one or two sentences, only \
when should_refine is true"}.\
"""


class RefineInterjectionSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description=(
            "Every `every_n_model_calls` model calls, and at most once per "
            "`cooldown_seconds`, ask a gate agent whether the turn so far is worth "
            "a refinement; run the refiner when it says so."
        ),
    )
    every_n_model_calls: int = Field(
        8, description="Model calls between two gate reviews within a turn."
    )
    cooldown_seconds: float = Field(
        120.0, description="Minimum time between two gate reviews."
    )
    gate_agent_identifier: Optional[str] = Field(
        None,
        description=(
            "Agent that decides whether to refine. Defaults to a fresh copy of "
            "the current agent; prefer a lean agent with tools and memory off."
        ),
    )
    gate_instructions: Optional[str] = Field(
        None, description="Override the gate instructions."
    )


class RefineInterjectionSource(InterjectionSource):
    """Prime's automatic `/refine`: after every N model calls, once the cooldown
    has passed, a cheap gate call over the transcript decides whether the turn
    so far is worth persisting; on yes the refiner runs with the gate's
    instructions and the agent is told what was kept. Nothing runs between
    checkpoints, and a rejected checkpoint costs one gate call."""

    source_name = "refine"
    source_description = "Asks the agent to refine its work every few steps."

    def __init__(
        self,
        refiner: Refiner,
        agent_creator: AgentCreator,
        enabled: bool,
        every_n_model_calls: int,
        cooldown_seconds: float,
        gate_agent_identifier: Optional[str],
        gate_instructions: Optional[str],
    ) -> None:
        self.enabled = enabled and bool(refiner.kinds)
        self.__refiner = refiner
        self.__agent_creator = agent_creator
        self.__every_n_model_calls = max(1, every_n_model_calls)
        self.__cooldown_seconds = cooldown_seconds
        self.__gate_agent_identifier = gate_agent_identifier
        self.__gate_instructions = gate_instructions or _DEFAULT_GATE_INSTRUCTIONS
        self.__model_calls = 0
        self.__last_review: Optional[float] = None

    async def after_model_call(self, context: InnerLoopContext) -> Optional[str]:
        self.__model_calls += 1
        if self.__model_calls % self.__every_n_model_calls != 0:
            return None
        now = time.monotonic()
        if (
            self.__last_review is not None
            and now - self.__last_review < self.__cooldown_seconds
        ):
            return None
        self.__last_review = now
        instructions = await self.__review(context)
        if instructions is None:
            return None
        return await self.__refiner.refine(
            focus=instructions, transcript=context.transcript, mode="auto"
        )

    async def __review(self, context: InnerLoopContext) -> Optional[str]:
        """The gate's instructions for the refiner when it approves, else None."""
        prompt = "\n\n".join(
            [
                self.__gate_instructions,
                f"<harness>\n{await self.__refiner.render_harness()}\n</harness>",
                "<transcript>\n"
                + self.__refiner.render_transcript(context.transcript)
                + "\n</transcript>",
            ]
        )
        try:
            reply = await run_sub_agent(
                agent_creator=self.__agent_creator,
                prompt=prompt,
                target_identifier=self.__gate_agent_identifier,
            )
        except Exception as e:
            logger.error(f"Refinement gate failed: {e}")
            return None
        verdict = parse_gate_reply(reply.content or "")
        if not verdict.get("should_refine"):
            return None
        return str(verdict.get("instructions") or "").strip()


def parse_gate_reply(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        verdict = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("Refinement gate returned no parseable JSON object.")
        return {}
    return verdict if isinstance(verdict, dict) else {}


class RefineInterjectionSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        refiner: Refiner,
        agent_creator: AgentCreator,
        refine_interjection_source_configuration: (
            RefineInterjectionSourceConfiguration
        ) = RefineInterjectionSourceConfiguration(),
    ) -> RefineInterjectionSource:
        configuration = refine_interjection_source_configuration
        return RefineInterjectionSource(
            refiner=refiner,
            agent_creator=agent_creator,
            enabled=configuration.enabled,
            every_n_model_calls=configuration.every_n_model_calls,
            cooldown_seconds=configuration.cooldown_seconds,
            gate_agent_identifier=configuration.gate_agent_identifier,
            gate_instructions=configuration.gate_instructions,
        )
