from dataclasses import dataclass
from typing import Optional

from zav.agents_sdk.domain.agent_creator import SubAgentApprovals
from zav.agents_sdk.domain.turn_inbox import TurnInbox


@dataclass(frozen=True)
class ToolApprovals:
    """How an agent settles calls that need approval. The root agent asks the
    user and shows the cards in its own output; a sub-agent follows its
    delegation's setting and, when it asks, publishes its cards for the root
    agent to show. Without an inbox (no interactive turn) nobody can be
    asked."""

    inbox: Optional[TurnInbox] = None
    mode: SubAgentApprovals = "ask"
    root: bool = True
