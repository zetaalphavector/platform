import os

from zav.logging import logger

from zav.agents_sdk import AgentSetupRetrieverFromFile, setup_app
from zav.agents_sdk.adapters import AgentDependencyRegistry
from zav.agents_sdk.adapters.local_agent_registries_factory import (
    LocalAgentRegistriesFactory,
)
from zav.agents_sdk.cli.load_chat_agent_factory import (
    from_string as import_chat_agent_class_registry_from_string,
)
from zav.agents_sdk.cli.local_mcp_oauth import setup_local_mcp_oauth
from zav.agents_sdk.controllers import routers as default_routers
from zav.agents_sdk.controllers.v1.mcp_server import mcp_server_router
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry
from zav.agents_sdk.domain.llm_configuration_store import LLMConfigurationStoreFromFile
from zav.agents_sdk.domain.mcp_server_setup import FileMCPServerSetupRetriever

zav_project_dir = os.environ["ZAV_PROJECT_DIR"]
zav_agent_setup_src = os.getenv("ZAV_AGENT_SETUP_SRC")
zav_secret_agent_setup_src = os.getenv("ZAV_SECRET_AGENT_SETUP_SRC")
openai_api_key = os.getenv("OPENAI_API_KEY")


import_chat_agent_class_registry_from_string(zav_project_dir)


def existing_path(path):
    return path if path and os.path.exists(path) else None


# A tools-only project (`za mcp init`) has no agent_setups.json, and agents
# projects may lack the env/ secrets file: the retriever treats a None path as
# "no setups from this file" instead of failing the boot. The store is anchored
# on the project dir (not the agent setups path) so tools-only projects still
# resolve named LLM configurations.
llm_configuration_store = LLMConfigurationStoreFromFile(
    llm_configurations_path=os.path.join(zav_project_dir, "llm_configurations.json"),
    secret_llm_configurations_path=os.path.join(
        zav_project_dir, "env", "llm_configurations.json"
    ),
)
agent_setup_retriever = AgentSetupRetrieverFromFile(
    file_path=existing_path(zav_agent_setup_src),
    secret_file_path=existing_path(zav_secret_agent_setup_src),
    llm_configuration_store=llm_configuration_store,
)
mcp_oauth_store, mcp_oauth_token_client = setup_local_mcp_oauth(AgentDependencyRegistry)

# Expose the agents-sdk tools over MCP only when serving via `za mcp` (which sets
# ZAV_SERVE_MCP). `za agents serve` stays agents-only and never mounts /mcp.
mcp_setup_src = os.getenv("ZAV_MCP_SETUP_SRC") or os.path.join(
    zav_project_dir, "mcp_setups.json"
)
_serve_mcp = os.getenv("ZAV_SERVE_MCP", "").strip().lower() in ("1", "true", "yes")
mcp_server_setup_retriever = (
    FileMCPServerSetupRetriever(mcp_setup_src)
    if _serve_mcp and os.path.exists(mcp_setup_src)
    else None
)
routers = list(default_routers)
if mcp_server_setup_retriever is not None:
    routers.append(("", mcp_server_router))

app = setup_app(
    agent_registries_factory=LocalAgentRegistriesFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=ChatAgentClassRegistry,
        agent_dependency_registry=AgentDependencyRegistry,
        # The retriever's layered store, not the bare file store: it also holds
        # the inline agent configurations registered under __agent__:<identifier>.
        llm_configuration_store=agent_setup_retriever.llm_configuration_store,
    ),
    debug_backend=logger.info,
    mcp_oauth_store=mcp_oauth_store,
    mcp_oauth_token_client=mcp_oauth_token_client,
    mcp_server_setup_retriever=mcp_server_setup_retriever,
    routers=routers,
)
