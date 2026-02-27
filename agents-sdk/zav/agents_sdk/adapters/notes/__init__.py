from zav.agents_sdk.adapters.notes.notes_service import (
    NotesService,
    NotesServiceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(NotesServiceFactory)

__all__ = [
    "NotesService",
    "NotesServiceFactory",
]
