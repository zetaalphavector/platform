from typing import Any, Dict, List, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryEntry,
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
        writable_source: Optional[str] = None,
        include_sources: Optional[Set[str]] = None,
        exclude_sources: Optional[Set[str]] = None,
    ):
        self.__stores = stores
        self.__enabled = enabled
        self.__inject_context = inject_context
        self.__provide_tools = provide_tools
        self.__read_instructions = read_instructions
        self.__write_instructions = write_instructions
        self.__include_sources = include_sources
        self.__exclude_sources = exclude_sources
        self.__writable_source_name = writable_source
        self.__writable_store: Optional[MemoryStore] = None
        self.__cached_active: Optional[List[MemoryStore]] = None

    def describe_loaded(self) -> Dict[str, Any]:
        writable = self.__get_writable_store()
        return {
            "enabled": self.__enabled,
            "sources": [s.source_name for s in self.__active_stores()],
            "writable": writable.source_name if writable else None,
        }

    async def to_prompt(self) -> str:
        if not self.__enabled:
            return ""

        sections: List[str] = []

        instructions = self.__read_instructions or _DEFAULT_MEMORY_INSTRUCTIONS
        if self.__write_instructions:
            instructions = f"{instructions}\n\n{self.__write_instructions}"
        sections.append(instructions)

        if self.__inject_context:
            entries = await self.__load_all_entries()
            if entries:
                sections.append(self.__format_entries(entries))

        return "\n\n".join(sections)

    async def get_tools(self) -> List[Tool]:
        if not self.__enabled or not self.__provide_tools:
            return []
        writable = self.__get_writable_store()
        if not writable:
            return []
        return writable.get_tools()

    async def __load_all_entries(self) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for store in self.__active_stores():
            try:
                entries.extend(await store.load_context())
            except Exception as e:
                logger.error(
                    f"Failed to load memory context from '{store.source_name}': {e}"
                )
        return entries

    def __format_entries(self, entries: List[MemoryEntry]) -> str:
        lines = ["## Recalled memories", ""]
        for entry in entries:
            parts = []
            if entry.id:
                parts.append(f"id:{entry.id}")
            if entry.created_at:
                parts.append(entry.created_at[:10])
            prefix = f"[{', '.join(parts)}] " if parts else ""
            lines.append(f"- {prefix}{entry.content}")
        return "\n".join(lines)

    def __active_stores(self) -> List[MemoryStore]:
        if self.__cached_active is not None:
            return self.__cached_active
        self.__cached_active = [
            store
            for store in self.__stores
            if is_source_active(
                store.source_name,
                store.enabled,
                self.__include_sources,
                self.__exclude_sources,
            )
        ]
        return self.__cached_active

    def __get_writable_store(self) -> Optional[MemoryStore]:
        if self.__writable_store is not None:
            return self.__writable_store
        active = self.__active_stores()
        if not active:
            return None
        if self.__writable_source_name:
            for store in active:
                if store.source_name == self.__writable_source_name:
                    self.__writable_store = store
                    return store
            return None
        self.__writable_store = active[0]
        return self.__writable_store


class MemoryProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        memory_store_group: MemoryStoreGroup = MemoryStoreGroup(items=[]),
        memory_provider_configuration: MemoryProviderConfiguration = (
            MemoryProviderConfiguration()
        ),
    ) -> MemoryProvider:
        return MemoryProvider(
            stores=memory_store_group.items,
            enabled=memory_provider_configuration.enabled,
            inject_context=memory_provider_configuration.inject_context,
            provide_tools=memory_provider_configuration.provide_tools,
            read_instructions=memory_provider_configuration.read_instructions,
            write_instructions=memory_provider_configuration.write_instructions,
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
