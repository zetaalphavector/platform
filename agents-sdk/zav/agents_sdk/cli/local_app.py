import os

from zav.logging import logger

from zav.agents_sdk import AgentSetupRetrieverFromFile, setup_app
from zav.agents_sdk.cli.load_chat_agent_factory import (
    from_string as import_chat_agent_factory_from_string,
)

zav_project_dir = os.environ["ZAV_PROJECT_DIR"]
zav_agent_setup_src = os.getenv("ZAV_AGENT_SETUP_SRC")
zav_secret_agent_setup_src = os.getenv("ZAV_SECRET_AGENT_SETUP_SRC")
openai_api_key = os.getenv("OPENAI_API_KEY")


chat_agent_factory = import_chat_agent_factory_from_string(zav_project_dir)

agent_setup_retriever = AgentSetupRetrieverFromFile(
    file_path=zav_agent_setup_src, secret_file_path=zav_secret_agent_setup_src
)

app = setup_app(
    agent_setup_retriever=agent_setup_retriever,
    chat_agent_factory=chat_agent_factory,
    debug_backend=logger.info,
)
