from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryEntry,
    MemoryStore,
    SearchableMemoryStore,
)
from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyFactory,
    DependencyGroup,
)


def _format_entries(entries: List[MemoryEntry]) -> str:
    lines = ["## Recalled memories", ""]
    for entry in entries:
        parts = []
        if entry.id:
            parts.append(f"id:{entry.id}")
        if entry.source_url:
            parts.append(f"source url: {entry.source_url}")
        if entry.created_at:
            parts.append(entry.created_at[:10])
        prefix = f"[{', '.join(parts)}] " if parts else ""
        lines.append(f"- {prefix}{entry.content}")
    return "\n".join(lines)


class MemoryContextStrategy(ABC):
    source_name: ClassVar[str]

    @abstractmethod
    async def to_prompt(self, stores: List[MemoryStore]) -> str:
        raise NotImplementedError


class FullMemoryContextStrategy(MemoryContextStrategy):
    source_name = "full"

    async def __load_all_entries(self, stores: List[MemoryStore]) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for store in stores:
            try:
                entries.extend(await store.load_context())
            except Exception as e:
                logger.error(
                    f"Failed to load memory context from '{store.source_name}': {e}"
                )
        return entries

    async def to_prompt(self, stores: List[MemoryStore]) -> str:
        entries = await self.__load_all_entries(stores)
        if not entries:
            return ""
        return _format_entries(entries)


class FullMemoryContextStrategyFactory(AgentDependencyFactory):
    @classmethod
    def create(cls) -> FullMemoryContextStrategy:
        return FullMemoryContextStrategy()


class RecentMemoryContextStrategyConfiguration(BaseModel):
    size: int = Field(
        10, description="Number of most recent memories to load into context."
    )


class RecentMemoryContextStrategy(MemoryContextStrategy):
    source_name = "recent"

    def __init__(self, size: int) -> None:
        self.__size = size

    async def to_prompt(self, stores: List[MemoryStore]) -> str:
        entries: List[MemoryEntry] = []
        for store in stores:
            try:
                recent_entries, total = await store.load_recent_context(self.__size)
                entries.extend(recent_entries)
            except Exception as e:
                logger.error(
                    f"Failed to load memory context from '{store.source_name}': {e}"
                )
        if not entries:
            return ""
        note = self.__pre_entries_note(stores, len(entries), total)
        formatted = _format_entries(entries)
        return f"{note}\n\n{formatted}" if note else formatted

    @staticmethod
    def __pre_entries_note(stores: List[MemoryStore], shown: int, total: int) -> str:
        if total <= shown:
            return ""
        note = (
            f"These are the {shown} most recent of {total} saved memories; "
            f"the other {total - shown} are not shown here."
        )
        if any(isinstance(store, SearchableMemoryStore) for store in stores):
            note += (
                "\n"
                "Before telling the user you don't know something about them, or "
                "before assuming that the stored memories are not relevant, "
                "use the search functionality to find any relevant memories "
                "before providing an answer. "
            )
        return note


class RecentMemoryContextStrategyFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        recent_memory_context_strategy_configuration: (
            RecentMemoryContextStrategyConfiguration
        ) = RecentMemoryContextStrategyConfiguration(),
    ) -> RecentMemoryContextStrategy:
        return RecentMemoryContextStrategy(
            size=recent_memory_context_strategy_configuration.size
        )


class MemoryContextStrategyGroup(DependencyGroup[MemoryContextStrategy]):
    __collects__ = MemoryContextStrategy


def pick_context_strategy(
    strategies: List[MemoryContextStrategy], name: str
) -> Optional[MemoryContextStrategy]:
    return next((s for s in strategies if s.source_name == name), None)
