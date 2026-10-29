from zav.agents_sdk.adapters.memory.memory_context import (
    FullMemoryContextStrategy,
    FullMemoryContextStrategyFactory,
    MemoryContextStrategy,
    MemoryContextStrategyGroup,
    RecentMemoryContextStrategy,
    RecentMemoryContextStrategyConfiguration,
    RecentMemoryContextStrategyFactory,
)
from zav.agents_sdk.adapters.memory.memory_provider import (
    MemoryProvider,
    MemoryProviderConfiguration,
    MemoryProviderFactory,
)
from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryCapability,
    MemoryEntry,
    MemoryReadError,
    MemorySaveError,
    MemoryStore,
    MemoryStoreGroup,
    MemoryUpdateError,
    SearchableMemoryStore,
)
from zav.agents_sdk.adapters.memory.stores.file_memory_store import (
    FileMemoryStore,
    FileMemoryStoreConfiguration,
    FileMemoryStoreFactory,
)
from zav.agents_sdk.adapters.memory.stores.in_memory_memory_store import (
    InMemoryMemoryStore,
    InMemoryMemoryStoreConfiguration,
    InMemoryMemoryStoreFactory,
)
from zav.agents_sdk.adapters.memory.stores.memory_note_memory_store import (
    MemoryNoteMemoryStore,
    MemoryNoteMemoryStoreConfiguration,
    MemoryNoteMemoryStoreFactory,
)
from zav.agents_sdk.adapters.memory.stores.searchable_memory_note_memory_store import (
    SearchableMemoryNoteMemoryStore,
    SearchableMemoryNoteMemoryStoreConfiguration,
    SearchableMemoryNoteMemoryStoreFactory,
)
from zav.agents_sdk.adapters.memory.stores.tag_note_memory_store import (
    TagNoteMemoryStore,
    TagNoteMemoryStoreConfiguration,
    TagNoteMemoryStoreFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(MemoryProviderFactory)
AgentDependencyRegistry.register(InMemoryMemoryStoreFactory)
AgentDependencyRegistry.register(FileMemoryStoreFactory)
AgentDependencyRegistry.register(TagNoteMemoryStoreFactory)
AgentDependencyRegistry.register(MemoryNoteMemoryStoreFactory)
AgentDependencyRegistry.register(SearchableMemoryNoteMemoryStoreFactory)
AgentDependencyRegistry.register(FullMemoryContextStrategyFactory)
AgentDependencyRegistry.register(RecentMemoryContextStrategyFactory)

__all__ = [
    "FileMemoryStore",
    "FileMemoryStoreConfiguration",
    "FileMemoryStoreFactory",
    "SearchableMemoryStore",
    "MemoryContextStrategy",
    "MemoryContextStrategyGroup",
    "FullMemoryContextStrategy",
    "FullMemoryContextStrategyFactory",
    "RecentMemoryContextStrategy",
    "RecentMemoryContextStrategyConfiguration",
    "RecentMemoryContextStrategyFactory",
    "InMemoryMemoryStore",
    "InMemoryMemoryStoreConfiguration",
    "InMemoryMemoryStoreFactory",
    "MemoryCapability",
    "MemoryProviderConfiguration",
    "MemoryEntry",
    "MemoryProvider",
    "MemoryProviderFactory",
    "MemoryReadError",
    "MemorySaveError",
    "MemoryStore",
    "MemoryStoreGroup",
    "MemoryUpdateError",
    "MemoryNoteMemoryStore",
    "MemoryNoteMemoryStoreConfiguration",
    "MemoryNoteMemoryStoreFactory",
    "SearchableMemoryNoteMemoryStore",
    "SearchableMemoryNoteMemoryStoreConfiguration",
    "SearchableMemoryNoteMemoryStoreFactory",
    "TagNoteMemoryStore",
    "TagNoteMemoryStoreConfiguration",
    "TagNoteMemoryStoreFactory",
]
