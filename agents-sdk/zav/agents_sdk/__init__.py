# flake8: noqa
#
# Lazy module: all public names are importable but only resolved on first access.
# This keeps CLI startup fast (~0.25s) by avoiding eager import of heavy adapter/domain
# modules (pandas, llm_tracing, etc.) that take ~2.4s to load.
#
# To add a new export: just add it to the TYPE_CHECKING block below.
# The lazy import mapping is derived automatically from that block.
#
import ast as _ast
import importlib as _importlib
import textwrap as _textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zav.agents_sdk.adapters.agent_setup_retrievers import (
        AgentSetupRetrieverFromFile,
        LocalAgentSetupRetriever,
    )
    from zav.agents_sdk.adapters.event_publishers import AbstractEventPublisher
    from zav.agents_sdk.domain.agent_code_bundle import AgentCodeBundle
    from zav.agents_sdk.domain.agent_creator import AgentCreator
    from zav.agents_sdk.domain.agent_dependency import (
        AgentDependencyFactory,
        AgentDependencyRegistry,
        AgentDependencyRegistryProtocol,
        DependencyGroup,
    )
    from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
    from zav.agents_sdk.domain.agent_setup_retriever import (
        AgentSetup,
        AgentSetupRetriever,
    )
    from zav.agents_sdk.domain.chat_agent import (
        ChatAgent,
        ProcessorAgent,
        StreamableChatAgent,
    )
    from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
    from zav.agents_sdk.domain.chat_agent_registry import (
        ChatAgentClassRegistry,
        ChatAgentClassRegistryProtocol,
    )
    from zav.agents_sdk.domain.chat_message import (
        ChatMessage,
        ChatMessageEvidence,
        ChatMessageSender,
        ContentPart,
        ContentPartTable,
        ContentPartTool,
        ConversationContext,
        CustomContext,
        CustomContextItem,
        DocumentContext,
        FilterContext,
        FunctionCallRequest,
        FunctionSpec,
        TagContext,
    )
    from zav.agents_sdk.domain.request_headers import RequestHeaders
    from zav.agents_sdk.domain.table import Table
    from zav.agents_sdk.domain.tools import (
        Tool,
        ToolsRegistry,
        ToolStreamingConfig,
        exclude_fields,
        hide,
        include_fields,
        streamable,
    )
    from zav.agents_sdk.setup_app import setup_app


def _build_lazy_imports() -> dict[str, str]:
    """Parse this file's TYPE_CHECKING block to build {name: module} mapping."""
    with open(__file__) as f:
        tree = _ast.parse(f.read())
    mapping: dict[str, str] = {}
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.If):
            continue
        test = node.test
        if not (isinstance(test, _ast.Name) and test.id == "TYPE_CHECKING"):
            continue
        for stmt in node.body:
            if isinstance(stmt, _ast.ImportFrom) and stmt.module:
                for alias in stmt.names:
                    mapping[alias.asname or alias.name] = stmt.module
    return mapping


_LAZY_IMPORTS = _build_lazy_imports()
__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = _importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'zav.agents_sdk' has no attribute {name!r}")


__all__ = [
    "AgentCreator",
    "AgentRegistriesFactory",
    "AgentCodeBundle",
    "AbstractEventPublisher",
    "AgentDependencyFactory",
    "AgentDependencyRegistry",
    "AgentDependencyRegistryProtocol",
    "DependencyGroup",
    "AgentSetup",
    "AgentSetupRetriever",
    "ChatAgent",
    "ChatAgentFactory",
    "ProcessorAgent",
    "StreamableChatAgent",
    "ChatAgentClassRegistry",
    "ChatAgentClassRegistryProtocol",
    "ChatMessage",
    "ChatMessageEvidence",
    "ChatMessageSender",
    "ContentPartTable",
    "ConversationContext",
    "DocumentContext",
    "FilterContext",
    "FunctionCallRequest",
    "FunctionSpec",
    "setup_app",
    "AgentSetupRetrieverFromFile",
    "LocalAgentSetupRetriever",
    "RequestHeaders",
    "StreamableChatAgent",
    "Table",
    "CustomContext",
    "CustomContextItem",
    "ContentPart",
    "ContentPartTool",
    "Tool",
    "ToolsRegistry",
    "ToolStreamingConfig",
    "exclude_fields",
    "hide",
    "include_fields",
    "streamable",
]
