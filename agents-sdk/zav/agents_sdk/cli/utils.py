import json
import os
from typing import Callable


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
        f.write(f"""from typing import AsyncGenerator, List

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
            stream=True,
        )
        async for chat_client_response in response:
            if chat_client_response.error is not None:
                raise chat_client_response.error
            if chat_client_response.chat_completion is None:
                raise Exception("No response from chat completion client")

            yield ChatMessage.from_orm(chat_client_response.chat_completion)
""")

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
                        "name": "gpt-5.4-mini",
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
        f.write(f"""from typing import Optional
from zav.agents_sdk import AgentDependencyFactory, AgentDependencyRegistry


class {class_name}:
    def __init__(self, api_url: Optional[str] = None):
        self.__api_url = api_url

    async def execute(self, input: str) -> str:
        # TODO: implement your dependency logic here
        return f"Received: {{input}}"


class {class_name}Factory(AgentDependencyFactory):
    @classmethod
    def create(cls, api_url: Optional[str] = None) -> {class_name}:
        return {class_name}(api_url=api_url)


AgentDependencyRegistry.register({class_name}Factory)
""")

    return dependency_name_snake


SOURCE_TEMPLATES = {
    "tools": {
        "base_import": "from zav.agents_sdk.adapters.tools.tools_source import ToolsSource",  # noqa: E501
        "extra_imports": "from zav.agents_sdk.domain.tools import Tool",
        "base_class": "ToolsSource",
        "body": '''
    source_name = "{source_name}"

    def __init__(self, {config_snake}_configuration: {class_name}Configuration):
        self.enabled = {config_snake}_configuration.enabled

    async def get_tools(self) -> list:
        async def example_tool(query: str) -> str:
            """An example tool. Replace with your own implementation.

            Args:
                query: Input to process.

            Returns:
                The processed result.
            """
            return f"Received: {{query}}"

        return [Tool.from_callable(name="{source_name}_example", executable=example_tool)]
''',  # noqa: E501
    },
    "instructions": {
        "base_import": "from zav.agents_sdk.adapters.instructions.instruction_source import InstructionSource",  # noqa: E501
        "extra_imports": "",
        "base_class": "InstructionSource",
        "body": """
    source_name = "{source_name}"

    def __init__(self, {config_snake}_configuration: {class_name}Configuration):
        self.enabled = {config_snake}_configuration.enabled

    async def to_prompt(self) -> str:
        # TODO: return your custom instructions
        return "You are a helpful assistant."
""",  # noqa: E501
    },
    "context": {
        "base_import": "from zav.agents_sdk.adapters.context.context_source import ContextSource",  # noqa: E501
        "extra_imports": "from zav.agents_sdk.adapters.context.context_source import ContextSourceDescription, ResolvedContextItem",  # noqa: E501
        "base_class": "ContextSource",
        "body": """
    source_name = "{source_name}"

    def __init__(self, {config_snake}_configuration: {class_name}Configuration):
        self.enabled = {config_snake}_configuration.enabled

    async def describe(self) -> ContextSourceDescription:
        return ContextSourceDescription(
            name="{source_name}",
            description="TODO: describe this context source",
        )

    async def resolve(self, query: str) -> list:
        # TODO: return relevant context items
        return []
""",  # noqa: E501
    },
    "processors": {
        "base_import": "from zav.agents_sdk.adapters.message_processing.message_processor import MessageProcessor, StreamItem",  # noqa: E501
        "extra_imports": "from typing import AsyncGenerator",
        "base_class": "MessageProcessor",
        "body": """
    source_name = "{source_name}"

    def __init__(self, {config_snake}_configuration: {class_name}Configuration):
        self.enabled = {config_snake}_configuration.enabled

    async def process_stream(
        self, stream: AsyncGenerator[StreamItem, None]
    ) -> AsyncGenerator[StreamItem, None]:
        async for response, message in stream:
            # TODO: transform the message before yielding
            yield response, message
""",  # noqa: E501
    },
}


def create_source_files(
    project_dir: str,
    dependencies_dir: str,
    source_name: str,
    provider_name: str,
    file_exists_callback: Callable[[str], str],
) -> str:
    class_name = to_camel_case(source_name)
    source_name_snake = to_snake_case(source_name)
    source_file = os.path.join(dependencies_dir, f"{source_name_snake}.py")
    while os.path.exists(source_file):
        source_name = file_exists_callback(source_name_snake)
        class_name = to_camel_case(source_name)
        source_name_snake = to_snake_case(source_name)
        source_file = os.path.join(dependencies_dir, f"{source_name_snake}.py")

    init_file = os.path.join(dependencies_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w"):
            pass

    project_init_file = os.path.join(project_dir, "__init__.py")
    with open(project_init_file, "a") as f:
        f.write(f"\nfrom .dependencies.{source_name_snake} import *\n")

    template = SOURCE_TEMPLATES.get(provider_name)
    if template is None:
        from zav.agents_sdk.cli.source_aliases import get_provider_source_bases

        base_info = get_provider_source_bases().get(provider_name, ("", "ABC"))
        base_module = base_info[0]
        base_class_name = base_info[1]
        base_import = (
            f"from {base_module} import {base_class_name}" if base_module else ""
        )
        extra_imports = ""
        base_class = base_class_name
        body = f"""
    source_name = "{source_name_snake}"

    def __init__(self, {source_name_snake}_configuration: {class_name}Configuration):
        self.enabled = {source_name_snake}_configuration.enabled
"""
    else:
        base_import = template["base_import"]
        extra_imports = template["extra_imports"]
        base_class = template["base_class"]
        body = template["body"].format(
            source_name=source_name_snake,
            class_name=class_name,
            config_snake=source_name_snake,
        )

    config_snake = source_name_snake

    imports = (
        "from zav.agents_sdk import AgentDependencyFactory, AgentDependencyRegistry"
    )
    if base_import:
        imports += f"\n{base_import}"
    if extra_imports:
        imports += f"\n{extra_imports}"

    with open(source_file, "w") as f:
        f.write(f"""from zav.pydantic_compat import BaseModel, Field
{imports}


class {class_name}Configuration(BaseModel):
    enabled: bool = Field(False, description="Whether this source is enabled.")


class {class_name}Source({base_class}):
{body}

class {class_name}SourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        {config_snake}_configuration: {class_name}Configuration = {class_name}Configuration(),
    ) -> {class_name}Source:
        return {class_name}Source({config_snake}_configuration={config_snake}_configuration)


AgentDependencyRegistry.register({class_name}SourceFactory)
""")  # noqa: E501  # noqa: E501

    return source_name_snake
