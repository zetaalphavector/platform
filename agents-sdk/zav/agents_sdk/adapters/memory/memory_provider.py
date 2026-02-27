from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

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

Apply a minimum signal gate: only save facts that would be \
genuinely valuable to recall in a future session. Do not save \
transient details, obvious information, or things already \
captured in existing memories.\
"""


class MemoryConfiguration(BaseModel):
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
    ):
        self.__stores = stores
        self.__enabled = enabled
        self.__inject_context = inject_context
        self.__provide_tools = provide_tools
        self.__read_instructions = read_instructions
        self.__write_instructions = write_instructions
        writable = self.__resolve_writable(stores, writable_source)
        self.__writable_store = writable

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
        if not self.__writable_store:
            return []
        return self.__writable_store.get_tools()

    async def __load_all_entries(self) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for store in self.__stores:
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
            lines.append(f"- {entry.content}")
        return "\n".join(lines)

    @staticmethod
    def __resolve_writable(
        stores: List[MemoryStore],
        writable_source: Optional[str],
    ) -> Optional[MemoryStore]:
        if not stores:
            return None
        if writable_source:
            for store in stores:
                if store.source_name == writable_source:
                    return store
            return None
        return stores[0]


class MemoryProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        memory_store_group: MemoryStoreGroup = MemoryStoreGroup(items=[]),
        memory_configuration: MemoryConfiguration = MemoryConfiguration(),
    ) -> MemoryProvider:
        return MemoryProvider(
            stores=memory_store_group.items,
            enabled=memory_configuration.enabled,
            inject_context=memory_configuration.inject_context,
            provide_tools=memory_configuration.provide_tools,
            read_instructions=memory_configuration.read_instructions,
            write_instructions=memory_configuration.write_instructions,
            writable_source=memory_configuration.writable_source,
        )
