from zav.agents_sdk.adapters.compaction.context_window_manager import (
    ContextWindowConfiguration,
    ContextWindowManager,
    ContextWindowManagerFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(ContextWindowManagerFactory)

__all__ = [
    "ContextWindowConfiguration",
    "ContextWindowManager",
    "ContextWindowManagerFactory",
]
