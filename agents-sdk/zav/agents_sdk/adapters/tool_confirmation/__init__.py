from zav.agents_sdk.adapters.tool_confirmation.session_approvals import (
    SessionApprovals,
    SessionApprovalsFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(SessionApprovalsFactory)

__all__ = ["SessionApprovals", "SessionApprovalsFactory"]
