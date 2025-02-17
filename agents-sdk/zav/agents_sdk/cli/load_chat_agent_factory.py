import importlib
import os
from inspect import isclass
from typing import Type

from zav.logging import logger

from zav.agents_sdk.domain.chat_agent import ChatAgent
from zav.agents_sdk.domain.chat_agent_registry import (
    ChatAgentClassRegistry,
    ChatAgentClassRegistryProtocol,
)


def from_string(zav_project_dir: str) -> ChatAgentClassRegistryProtocol:
    import_str = zav_project_dir.replace("/", ".")
    module_str, _, attrs_str = import_str.partition(":")
    if not module_str:
        raise Exception(
            f"Import string {import_str} must be in format <module> "
            "or <module>:<attribute>."
        )
    try:
        if zav_project_dir == os.getcwd():
            import sys

            sys.path.append("..")
            dynamic_module = importlib.import_module(os.path.basename(zav_project_dir))
        else:
            import sys

            sys.path.append(".")
            dynamic_module = importlib.import_module(module_str)
    except ImportError as exc:
        if exc.name != module_str:
            raise exc from None
        raise Exception(f"Could not import module {module_str}.")
    instance = dynamic_module
    chat_agent_class_registry = getattr(instance, "ChatAgentClassRegistry", None)
    if not attrs_str:
        # Look for a class named ChatAgentClassRegistry in the module
        if not chat_agent_class_registry:
            raise Exception(
                f"Module {module_str} does not have a ChatAgentClassRegistry class."
            )
    else:
        # Look for the attribute in the module
        try:
            for attr_str in attrs_str.split("."):
                instance = getattr(instance, attr_str)
        except AttributeError:
            raise Exception(
                f'Attribute "{attrs_str}" not found in module "{module_str}".'
            )

        if isclass(instance):
            if issubclass(instance, ChatAgentClassRegistry):
                chat_agent_class_registry = instance
            elif issubclass(instance, ChatAgent):
                chat_agent: Type[ChatAgent] = instance
                chat_agent_class_registry = (
                    chat_agent_class_registry or ChatAgentClassRegistry
                )
                # Make sure the chat_agent is registered with the factory
                chat_agent_class_registry.register()(chat_agent)

    if not chat_agent_class_registry:
        raise Exception(
            f"Attribute {attrs_str} in module {module_str} is not a ChatAgent"
            " or ChatAgentClassRegistry."
        )
    logger.info(
        f"Loaded {len(chat_agent_class_registry.registry)} chat agents from "
        f"{import_str}"
    )
    return chat_agent_class_registry
