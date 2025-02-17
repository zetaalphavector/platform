from typing import Any, Dict

from pydantic import BaseModel


class AgentEvent(BaseModel):
    sender_agent_identifier: str
    recipient_agent_identifier: str
    payload: Dict[str, Any]
