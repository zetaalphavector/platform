import json
import re
from typing import List, Literal, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel

from zav.agents_sdk.adapters.agent_delegation.agent_delegation_provider import (
    AgentDelegationProvider,
)
from zav.agents_sdk.adapters.instructions.instructions_provider import (
    InstructionsProvider,
)
from zav.agents_sdk.adapters.memory.memory_provider import MemoryProvider
from zav.agents_sdk.adapters.skills.skills_provider import SkillsProvider
from zav.agents_sdk.adapters.skills.skills_source import SkillsSource

EditKind = Literal["memory", "skill", "prompt_note", "sub_agent"]

_NAME_PATTERN = re.compile(r"[^a-z0-9]+")


class RefinementEdit(BaseModel):
    """One typed change to the harness, with the evidence that motivated it."""

    kind: EditKind
    name: Optional[str] = None
    description: Optional[str] = None
    content: str
    tools: Optional[List[str]] = None
    trigger: str
    intended_effect: str


class RefinementApplyError(Exception):
    pass


def available_kinds(
    memory_provider: MemoryProvider,
    skills_provider: SkillsProvider,
    instructions_provider: InstructionsProvider,
    agent_delegation_provider: AgentDelegationProvider,
) -> List[EditKind]:
    """The edit kinds the agent's harness can take: one per store that is on
    and has somewhere to write."""
    kinds: List[EditKind] = []
    if memory_provider.writable_store is not None:
        kinds.append("memory")
    if skills_provider.writable_source is not None:
        kinds.append("skill")
    if instructions_provider.writable_source is not None:
        kinds.append("prompt_note")
    if agent_delegation_provider.writable_source is not None:
        kinds.append("sub_agent")
    return kinds


def parse_edits(text: str) -> List[RefinementEdit]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        logger.warning(f"Refiner returned no parseable JSON array: {e}")
        return []
    edits: List[RefinementEdit] = []
    for item in items if isinstance(items, list) else []:
        try:
            edits.append(RefinementEdit.parse_obj(item))
        except Exception as e:
            logger.warning(f"Skipping malformed refinement edit {item!r}: {e}")
    return edits


def slug(name: str) -> str:
    return _NAME_PATTERN.sub("-", name.strip().lower()).strip("-")


def skill_markdown(name: str, description: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body.strip()}\n"


async def apply_edit(
    edit: RefinementEdit,
    memory_provider: MemoryProvider,
    skills_provider: SkillsProvider,
    instructions_provider: InstructionsProvider,
    agent_delegation_provider: AgentDelegationProvider,
) -> str:
    """Write one edit to the store that owns its kind. Returns a short
    description of what was written; raises RefinementApplyError otherwise."""
    if not edit.content.strip():
        raise RefinementApplyError("empty content")
    if edit.kind == "memory":
        if memory_provider.writable_store is None:
            raise RefinementApplyError("memory is not writable")
        entry = await memory_provider.writable_store.save(edit.content)
        return f"memory {entry.id}"
    name = slug(edit.name or "")
    if not name:
        raise RefinementApplyError(f"{edit.kind} needs a name")
    if edit.kind == "skill":
        if skills_provider.writable_source is None:
            raise RefinementApplyError("skills are not writable")
        content = skill_markdown(name, edit.description or edit.trigger, edit.content)
        SkillsSource.parse_skill_properties(content)
        await skills_provider.writable_source.create_skill(name, content)
        return f"skill {name}"
    if edit.kind == "sub_agent":
        source = agent_delegation_provider.writable_source
        if source is None:
            raise RefinementApplyError("sub-agents are not writable")
        tools = edit.tools or []
        unavailable = sorted(set(tools) - set(source.available_tools))
        if unavailable:
            raise RefinementApplyError(
                f"tool sources not available to sub-agents: {', '.join(unavailable)}"
            )
        await source.create_agent(
            name, edit.description or edit.trigger, edit.content, tools
        )
        return f"sub-agent {name}"
    if instructions_provider.writable_source is None:
        raise RefinementApplyError("instructions are not writable")
    await instructions_provider.writable_source.write_note(name, edit.content)
    return f"prompt note {name}"
