from zav.pydantic_compat import BaseModel


class AgentPosition(BaseModel):
    """The agent a dependency resolution builds for. Each resolution of a turn,
    the root agent's and every sub-agent's, gets its own; a dependency typed
    with it receives that agent's."""

    agent_identifier: str
