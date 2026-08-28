import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from zav.pydantic_compat import BaseModel, Field


class MCPServerSetupNotFound(Exception):
    """Addressing is strict: a request must name an existing setup. Raised by
    the command handlers; the transport adapter maps it to INVALID_PARAMS."""


class MCPServerSetup(BaseModel):
    """Configuration for one MCP server exposure: a curated tool bundle.

    ``configuration`` reuses the same blocks as an agent's ``agent_configuration``
    (``tools_provider_configuration``, the per-source ``*_source_configuration``
    blocks, and policy blocks such as ``llm_selection_configuration``), but this
    is a tool bundle, not an agent: no instructions or agent-loop fields.
    """

    mcp_server_identifier: str
    configuration: Dict[str, Any] = Field(default_factory=dict)


class MCPServerSetupRetriever(ABC):
    # Tenant-parameterized: hosted retrievers resolve per-tenant setups; local
    # (file/in-memory) retrievers ignore the tenant and serve one config.
    @abstractmethod
    async def get(
        self, tenant: str, mcp_server_identifier: str
    ) -> Optional[MCPServerSetup]:
        raise NotImplementedError

    async def list(self, tenant: str) -> List[MCPServerSetup]:
        return []


class FileMCPServerSetupRetriever(MCPServerSetupRetriever):
    def __init__(self, file_path: str):
        self.__file_path = file_path

    def __load(self) -> Dict[str, MCPServerSetup]:
        # Re-read on every request so local edits apply without a restart.
        path = Path(self.__file_path)
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed MCP setups file {path}: {error}") from error
        setups = [MCPServerSetup(**item) for item in raw]
        return {setup.mcp_server_identifier: setup for setup in setups}

    async def get(
        self, tenant: str, mcp_server_identifier: str
    ) -> Optional[MCPServerSetup]:
        return self.__load().get(mcp_server_identifier)

    async def list(self, tenant: str) -> List[MCPServerSetup]:
        return list(self.__load().values())
