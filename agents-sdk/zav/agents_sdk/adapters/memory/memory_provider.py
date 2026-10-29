from typing import Any, Dict, List, Optional, Set

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import active_sources, pick_writable
from zav.agents_sdk.adapters.memory.memory_context import (
    MemoryContextStrategy,
    MemoryContextStrategyGroup,
    pick_context_strategy,
)
from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryStore,
    MemoryStoreGroup,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool

_DEFAULT_MEMORY_INSTRUCTIONS = """\
## Memory

You have access to a persistent memory system. Recalled memories \
from previous conversations are shown below when available.

MEMORY PROTOCOL:
1. At the start of each conversation, review any recalled memories \
shown below. They contain important context from prior sessions.
2. As you work, actively save important discoveries using the \
`save_memory` tool — user corrections, preferences, project \
conventions, and key decisions.
3. When the user shares personal details (name, role, preferences, \
expertise), ALWAYS save them. Building a user profile is critical \
for personalization across sessions.
4. When you learn that a previously saved memory is outdated, \
incorrect, or needs refinement, use the `update_memory` tool \
to replace it. Provide the memory's internal id shown in the recalled \
memories list and the full updated content.

Apply a minimum signal gate: only save facts that would be \
genuinely valuable to recall in a future session. Do not save \
transient details, obvious information, or things already \
captured in existing memories.\
"""


class MemoryProviderConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description="Enable the memory system.",
    )
    inject_context: bool = Field(
        True,
        description=(
            "Load and inject relevant memories into the system prompt each turn."
        ),
    )
    memory_context_strategy: str = Field(
        "full",
        description="source_name of the memory context strategy to load.",
    )
    provide_tools: bool = Field(
        True,
        description="Expose memory write tools to the Agent.",
    )
    writable_source: Optional[str] = Field(
        None,
        description=(
            "source_name of the store to write to. "
            "When None, the first registered store is used."
        ),
    )
    read_instructions: Optional[str] = Field(
        None,
        description=(
            "Override the default memory instructions. "
            "When None, the built-in instructions are used."
        ),
    )
    write_instructions: Optional[str] = Field(
        None,
        description=(
            "Additional instructions appended after the default, "
            "about when and how to write to memory."
        ),
    )
    include_sources: Optional[List[str]] = Field(
        None, description="Allowlist of memory store source names to include."
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of memory store source names to exclude."
    )


class MemoryProvider:
    def __init__(
        self,
        stores: List[MemoryStore],
        enabled: bool,
        inject_context: bool,
        provide_tools: bool,
        read_instructions: Optional[str],
        write_instructions: Optional[str],
        context_strategies: List[MemoryContextStrategy],
        memory_context_strategy: str = "full",
        writable_source: Optional[str] = None,
        include_sources: Optional[Set[str]] = None,
        exclude_sources: Optional[Set[str]] = None,
    ):
        self.__enabled = enabled
        self.__inject_context = inject_context
        self.__provide_tools = provide_tools
        self.__read_instructions = read_instructions
        self.__write_instructions = write_instructions
        self.__active = active_sources(
            stores, enabled, include_sources, exclude_sources
        )
        self.__writable = pick_writable(self.__active, writable_source)
        self.__context_strategy = pick_context_strategy(
            context_strategies, memory_context_strategy
        )

    @property
    def active_stores(self) -> List[MemoryStore]:
        return self.__active

    @property
    def writable_store(self) -> Optional[MemoryStore]:
        """Where memories are written, for the agent's tools and for anything
        else in the build that persists memories on its behalf. None when
        memory is off or no store qualifies."""
        return self.__writable

    def describe_loaded(self) -> Dict[str, Any]:
        return {
            "enabled": self.__enabled,
            "sources": [s.source_name for s in self.__active],
            "writable": self.__writable.source_name if self.__writable else None,
        }

    async def to_prompt(self) -> str:
        if not self.__enabled:
            return ""

        sections: List[str] = []

        instructions = self.__read_instructions or _DEFAULT_MEMORY_INSTRUCTIONS
        if self.__write_instructions:
            instructions = f"{instructions}\n\n{self.__write_instructions}"
        sections.append(instructions)

        if self.__inject_context and self.__context_strategy is not None:
            context = await self.__context_strategy.to_prompt(self.__active)
            if context:
                sections.append(context)

        return "\n\n".join(sections)

    async def get_tools(self) -> List[Tool]:
        if not self.__enabled or not self.__provide_tools or not self.__writable:
            return []
        return self.__writable.get_tools()


class MemoryProviderFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls,
        memory_store_group: MemoryStoreGroup = MemoryStoreGroup(items=[]),
        memory_context_strategy_group: MemoryContextStrategyGroup = (
            MemoryContextStrategyGroup(items=[])
        ),
        memory_provider_configuration: MemoryProviderConfiguration = (
            MemoryProviderConfiguration()
        ),
    ) -> MemoryProvider:
        return MemoryProvider(
            stores=memory_store_group.items,
            context_strategies=memory_context_strategy_group.items,
            enabled=memory_provider_configuration.enabled,
            inject_context=memory_provider_configuration.inject_context,
            provide_tools=memory_provider_configuration.provide_tools,
            read_instructions=memory_provider_configuration.read_instructions,
            write_instructions=memory_provider_configuration.write_instructions,
            memory_context_strategy=(
                memory_provider_configuration.memory_context_strategy
            ),
            writable_source=memory_provider_configuration.writable_source,
            include_sources=(
                set(memory_provider_configuration.include_sources)
                if memory_provider_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(memory_provider_configuration.exclude_sources)
                if memory_provider_configuration.exclude_sources
                else None
            ),
        )
