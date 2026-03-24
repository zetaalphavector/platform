from zav.agents_sdk.adapters.memory.file_memory_store import (
    FileMemoryStore,
    FileMemoryStoreConfig,
    FileMemoryStoreFactory,
)
from zav.agents_sdk.adapters.memory.in_memory_memory_store import (
    InMemoryMemoryStore,
    InMemoryMemoryStoreConfig,
    InMemoryMemoryStoreFactory,
)
from zav.agents_sdk.adapters.memory.memory_provider import (
    MemoryConfiguration,
    MemoryProvider,
    MemoryProviderFactory,
)
from zav.agents_sdk.adapters.memory.memory_store import (
    MemoryCapability,
    MemoryEntry,
    MemoryStore,
    MemoryStoreGroup,
)
from zav.agents_sdk.adapters.memory.tag_note_memory_store import (
    TagNoteMemoryStore,
    TagNoteMemoryStoreConfig,
    TagNoteMemoryStoreFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(MemoryProviderFactory)
AgentDependencyRegistry.register(InMemoryMemoryStoreFactory)
AgentDependencyRegistry.register(FileMemoryStoreFactory)
AgentDependencyRegistry.register(TagNoteMemoryStoreFactory)

__all__ = [
    "FileMemoryStore",
    "FileMemoryStoreConfig",
    "FileMemoryStoreFactory",
    "InMemoryMemoryStore",
    "InMemoryMemoryStoreConfig",
    "InMemoryMemoryStoreFactory",
    "MemoryCapability",
    "MemoryConfiguration",
    "MemoryEntry",
    "MemoryProvider",
    "MemoryProviderFactory",
    "MemoryStore",
    "MemoryStoreGroup",
    "TagNoteMemoryStore",
    "TagNoteMemoryStoreConfig",
    "TagNoteMemoryStoreFactory",
]
