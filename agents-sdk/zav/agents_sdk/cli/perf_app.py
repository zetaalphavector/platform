import asyncio
import os

from zav.logging import logger

from zav.agents_sdk import AgentSetupRetrieverFromFile, setup_app
from zav.agents_sdk.adapters import AgentDependencyRegistry
from zav.agents_sdk.adapters.agent_state import LocalFileChatAgentStateStore
from zav.agents_sdk.adapters.local_agent_registries_factory import (
    LocalAgentRegistriesFactory,
)
from zav.agents_sdk.adapters.stream_buffer import (
    LocalFileChatStreamRecordingStore,
    LocalFileChatStreamSupervisionStore,
)
from zav.agents_sdk.cli.load_chat_agent_factory import (
    from_string as import_chat_agent_class_registry_from_string,
)
from zav.agents_sdk.cli.local_mcp_oauth import setup_local_mcp_oauth
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry

zav_project_dir = os.environ["ZAV_PROJECT_DIR"]
zav_agent_setup_src = os.getenv("ZAV_AGENT_SETUP_SRC")
zav_secret_agent_setup_src = os.getenv("ZAV_SECRET_AGENT_SETUP_SRC")
openai_api_key = os.getenv("OPENAI_API_KEY")
storage_path = os.environ["STORAGE_PATH"]
perf_label = os.getenv("ZAV_PERF_LABEL", "server")


import_chat_agent_class_registry_from_string(zav_project_dir)

agent_setup_retriever = AgentSetupRetrieverFromFile(
    file_path=zav_agent_setup_src, secret_file_path=zav_secret_agent_setup_src
)
mcp_oauth_store, mcp_oauth_token_client = setup_local_mcp_oauth(AgentDependencyRegistry)

# All perf servers share one on-disk state store so a turn checkpointed by one
# server can be resumed by another. This is the file-backed stand-in for the
# cross-pod state table the production registry will use.
agent_state_store = LocalFileChatAgentStateStore(
    base_path=os.path.join(storage_path, "agent-traces", "state")
)

# Likewise, all perf servers share one on-disk stream recording, so a turn
# produced on one server can be replayed by another when a client reconnects to
# the "wrong" pod. This is the file-backed stand-in for the durable stream table.
chat_stream_recording_store = LocalFileChatStreamRecordingStore(
    base_path=os.path.join(storage_path, "agent-traces", "recording")
)

# The supervision row lives next to the recording in the shared dir, so a
# reconnecting server reads the producing server's heartbeat and status and can
# tell a turn that is still being produced apart from one whose server died.
chat_stream_supervision_store = LocalFileChatStreamSupervisionStore(
    base_path=os.path.join(storage_path, "agent-traces", "supervision")
)

app = setup_app(
    agent_registries_factory=LocalAgentRegistriesFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=ChatAgentClassRegistry,
        agent_dependency_registry=AgentDependencyRegistry,
    ),
    debug_backend=logger.info,
    mcp_oauth_store=mcp_oauth_store,
    mcp_oauth_token_client=mcp_oauth_token_client,
    agent_state_store=agent_state_store,
    chat_stream_recording_store=chat_stream_recording_store,
    chat_stream_supervision_store=chat_stream_supervision_store,
)


@app.get("/perf/info")
async def perf_info():
    return {"label": perf_label, "pid": os.getpid()}


if os.getenv("ZAV_PERF_MODE") == "1":

    @app.post("/perf/kill")
    async def perf_kill():
        # Simulate an abrupt process death (e.g. a Kubernetes pod OOMKill):
        # exit the process WITHOUT unwinding the stack, so the in-flight turn's
        # try/finally end-of-turn save never runs and only the periodic
        # checkpoint on disk survives. That partial checkpoint is exactly what
        # resume must recover from, so a graceful shutdown here would hide the
        # behaviour we test.
        async def __die() -> None:
            await asyncio.sleep(0.1)
            os._exit(137)

        asyncio.create_task(__die())
        return {"status": "killing", "label": perf_label, "pid": os.getpid()}
