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
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry

zav_project_dir = os.environ["ZAV_PROJECT_DIR"]
zav_agent_setup_src = os.getenv("ZAV_AGENT_SETUP_SRC")
zav_secret_agent_setup_src = os.getenv("ZAV_SECRET_AGENT_SETUP_SRC")
openai_api_key = os.getenv("OPENAI_API_KEY")


import_chat_agent_class_registry_from_string(zav_project_dir)

agent_setup_retriever = AgentSetupRetrieverFromFile(
    file_path=zav_agent_setup_src, secret_file_path=zav_secret_agent_setup_src
)

app = setup_app(
    agent_registries_factory=LocalAgentRegistriesFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=ChatAgentClassRegistry,
        agent_dependency_registry=AgentDependencyRegistry,
    ),
    debug_backend=logger.info,
)
