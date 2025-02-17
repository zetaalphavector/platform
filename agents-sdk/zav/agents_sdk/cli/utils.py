import json
import os
from typing import Callable, Optional


def is_valid_project_directory(directory: Optional[str] = None) -> bool:
    if directory is None:
        return False
    init_file = os.path.join(directory, "__init__.py")
    if not os.path.isfile(init_file):
        return False
    with open(init_file, "r") as f:
        content = f.read()
        return "Zeta Alpha Agents SDK" in content


def to_camel_case(name: str) -> str:
    return "".join(
        word.capitalize() for word in name.replace("-", " ").replace("_", " ").split()
    )


def to_snake_case(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").replace(" ", "_").lower()


def get_default_openai_key(env_agent_setups_file: str) -> str:

    default_openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if os.path.exists(env_agent_setups_file):
        with open(env_agent_setups_file, "r") as f:
            env_setups = json.load(f)
            for setup in env_setups:
                if "openai_api_key" in setup.get("llm_client_configuration", {}).get(
                    "vendor_configuration", {}
                ).get("openai", {}):
                    default_openai_api_key = setup["llm_client_configuration"][
                        "vendor_configuration"
                    ]["openai"]["openai_api_key"]
                    break

    return default_openai_api_key


def init_project(project_dir: str, sdk_version: str):
    os.makedirs(project_dir, exist_ok=True)

    # Create .gitignore file
    with open(os.path.join(project_dir, ".gitignore"), "w") as f:
        f.write("env/\n")

    # Create __init__.py file
    with open(os.path.join(project_dir, "__init__.py"), "w") as f:
        f.write(
            f'"""\nGenerated using Zeta Alpha Agents SDK Version: {sdk_version}\n"""\n'
        )

    # Create agent_setups.json file
    with open(os.path.join(project_dir, "agent_setups.json"), "w") as f:
        json.dump([], f)

    # Create env/agent_setups.json file
    env_project_dir = os.path.join(project_dir, "env")
    os.makedirs(env_project_dir, exist_ok=True)
    with open(os.path.join(env_project_dir, "agent_setups.json"), "w") as f:
        json.dump([], f)


def init_dependencies(project_dir: str):
    dependencies_dir = os.path.join(project_dir, "dependencies")
    os.makedirs(dependencies_dir, exist_ok=True)
    return dependencies_dir


def create_agent_files(
    project_dir: str,
    agent_name: str,
    agent_exists_callback: Callable[[str], str],
    openai_key_prompt_callback: Callable[[str], str],
):
    class_name = to_camel_case(agent_name)
    agent_name_snake = to_snake_case(agent_name)
    agent_file = os.path.join(project_dir, f"{agent_name_snake}.py")
    while os.path.exists(agent_file):
        agent_name = agent_exists_callback(agent_name_snake)
        class_name = to_camel_case(agent_name)
        agent_name_snake = to_snake_case(agent_name)
        agent_file = os.path.join(project_dir, f"{agent_name_snake}.py")

    # Check for existing OpenAI API key in env/agent_setups.json
    env_agent_setups_file = os.path.join(project_dir, "env", "agent_setups.json")
    default_openai_api_key = get_default_openai_key(env_agent_setups_file)

    # Ask for OpenAI API key
    default_openai_api_key_obscured = default_openai_api_key[:8] + "..."
    openai_api_key = openai_key_prompt_callback(default_openai_api_key_obscured)
    if openai_api_key == default_openai_api_key_obscured:
        openai_api_key = default_openai_api_key

    # Append import statement to __init__.py
    init_file = os.path.join(project_dir, "__init__.py")
    with open(init_file, "a") as f:
        f.write(f"\nfrom .{agent_name_snake} import *\n")

    # Create the agent file
    with open(agent_file, "w") as f:
        f.write(
            f"""from typing import AsyncGenerator, List

from zav.agents_sdk import ChatAgentClassRegistry, ChatMessage, StreamableChatAgent
from zav.agents_sdk.adapters import ZAVChatCompletionClient


@ChatAgentClassRegistry.register()
class {class_name}(StreamableChatAgent):
    agent_name = "{agent_name_snake}"

    def __init__(self, client: ZAVChatCompletionClient):
        self.client = client

    async def execute_streaming(
        self, conversation: List[ChatMessage]
    ) -> AsyncGenerator[ChatMessage, None]:
        response = await self.client.complete(
            messages=conversation,
            max_tokens=2048,
            stream=True,
        )
        async for chat_client_response in response:
            if chat_client_response.error is not None:
                raise chat_client_response.error
            if chat_client_response.chat_completion is None:
                raise Exception("No response from chat completion client")

            yield ChatMessage.from_orm(chat_client_response.chat_completion)
"""
        )

    # Update agent_setups.json
    agent_setups_file = os.path.join(project_dir, "agent_setups.json")
    with open(agent_setups_file, "r+") as f:
        setups = json.load(f)
        setups.append(
            {
                "agent_identifier": agent_name_snake,
                "agent_name": agent_name_snake,
                "llm_client_configuration": {
                    "vendor": "openai",
                    "vendor_configuration": {},
                    "model_configuration": {
                        "name": "gpt-4o-mini",
                        "type": "chat",
                        "temperature": 0.0,
                    },
                },
            }
        )
        f.seek(0)
        json.dump(setups, f, indent=2)
        f.truncate()

    # Update env/agent_setups.json
    with open(env_agent_setups_file, "r+") as f:
        env_setups = json.load(f)
        env_setups.append(
            {
                "agent_identifier": agent_name_snake,
                "llm_client_configuration": {
                    "vendor_configuration": {
                        "openai": {"openai_api_key": openai_api_key, "openai_org": ""}
                    }
                },
            }
        )
        f.seek(0)
        json.dump(env_setups, f, indent=2)
        f.truncate()

    return agent_name_snake


def create_dependency_files(
    project_dir: str,
    dependencies_dir: str,
    dependency_name: str,
    dependency_file_exists_callback: Callable[[str], str],
    # class_name: str,
    # dependency_name_snake: str,
    # dependency_file: str,
):
    class_name = to_camel_case(dependency_name)
    dependency_name_snake = to_snake_case(dependency_name)
    dependency_file = os.path.join(dependencies_dir, f"{dependency_name_snake}.py")
    while os.path.exists(dependency_file):
        dependency_name = dependency_file_exists_callback(dependency_name_snake)
        class_name = to_camel_case(dependency_name)
        dependency_name_snake = to_snake_case(dependency_name)
        dependency_file = os.path.join(dependencies_dir, f"{dependency_name_snake}.py")

    # Ensure __init__.py exists in dependencies directory
    init_file = os.path.join(dependencies_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w"):
            pass

    # Append import statement to project __init__.py
    project_init_file = os.path.join(project_dir, "__init__.py")
    with open(project_init_file, "a") as f:
        f.write(f"\nfrom .dependencies.{dependency_name_snake} import *\n")

    # Create the dependency file
    with open(dependency_file, "w") as f:
        f.write(
            f"""from typing import Dict, Optional
import httpx
from zav.agents_sdk import AgentDependencyFactory, AgentDependencyRegistry


class {class_name}:
    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self.headers = headers

    async def crawl(self, url: str) -> str:
        \"\"\"Crawl the given URL and return the HTML content.\"\"\"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            return response.text


class {class_name}Factory(AgentDependencyFactory):
    @classmethod
    def create(cls, headers: Optional[Dict[str, str]] = None) -> {class_name}:
        return {class_name}(headers=headers)


AgentDependencyRegistry.register({class_name}Factory)
"""
        )

    return dependency_name_snake
