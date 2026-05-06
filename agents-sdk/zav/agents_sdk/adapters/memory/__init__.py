from zav.agents_sdk.adapters.memory.memory_provider import (
    MemoryProvider,
    MemoryProviderConfiguration,
    MemoryProviderFactory,
)
from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryCapability,
    MemoryEntry,
    MemorySaveError,
    MemoryStore,
    MemoryStoreGroup,
    MemoryUpdateError,
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

__all__ = [
    "FileMemoryStore",
    "FileMemoryStoreConfiguration",
    "FileMemoryStoreFactory",
    "InMemoryMemoryStore",
    "InMemoryMemoryStoreConfiguration",
    "InMemoryMemoryStoreFactory",
    "MemoryCapability",
    "MemoryProviderConfiguration",
    "MemoryEntry",
    "MemoryProvider",
    "MemoryProviderFactory",
    "MemorySaveError",
    "MemoryStore",
    "MemoryStoreGroup",
    "MemoryUpdateError",
    "MemoryNoteMemoryStore",
    "MemoryNoteMemoryStoreConfiguration",
    "MemoryNoteMemoryStoreFactory",
    "TagNoteMemoryStore",
    "TagNoteMemoryStoreConfiguration",
    "TagNoteMemoryStoreFactory",
]
