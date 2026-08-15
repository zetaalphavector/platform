import inspect
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from zav.pydantic_compat import BaseModel


def _camel_to_snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    s = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", s)
    return s.lower()


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    config_key: str
    display_name: str
    cli_name: str
    source_base: Optional[Tuple[str, str]] = None
    config_model: Optional[Tuple[str, str]] = None


_PROVIDERS: Tuple[ProviderInfo, ...] = (
    ProviderInfo(
        name="tools",
        config_key="tools_provider_configuration",
        display_name="Tools",
        cli_name="tool-sets",
        source_base=("zav.agents_sdk.adapters.tools.tools_provider", "ToolsSource"),
        config_model=(
            "zav.agents_sdk.adapters.tools.tools_provider",
            "ToolsProviderConfiguration",
        ),
    ),
    ProviderInfo(
        name="skills",
        config_key="skills_provider_configuration",
        display_name="Skills",
        cli_name="skill-sources",
        source_base=("zav.agents_sdk.adapters.skills.skills_provider", "SkillsSource"),
        config_model=(
            "zav.agents_sdk.adapters.skills.skills_provider",
            "SkillsProviderConfiguration",
        ),
    ),
    ProviderInfo(
        name="memory",
        config_key="memory_provider_configuration",
        display_name="Memory",
        cli_name="memory-stores",
        source_base=("zav.agents_sdk.adapters.memory.memory_provider", "MemoryStore"),
        config_model=(
            "zav.agents_sdk.adapters.memory.memory_provider",
            "MemoryProviderConfiguration",
        ),
    ),
    ProviderInfo(
        name="context",
        config_key="context_provider_configuration",
        display_name="Context",
        cli_name="context-sources",
        source_base=(
            "zav.agents_sdk.adapters.context.context_provider",
            "ContextSource",
        ),
        config_model=(
            "zav.agents_sdk.adapters.context.context_provider",
            "ContextProviderConfiguration",
        ),
    ),
    ProviderInfo(
        name="instructions",
        config_key="instructions_provider_configuration",
        display_name="Instructions",
        cli_name="instruction-sources",
        source_base=(
            "zav.agents_sdk.adapters.instructions.instructions_provider",
            "InstructionSource",
        ),
        config_model=(
            "zav.agents_sdk.adapters.instructions.instructions_provider",
            "InstructionsProviderConfiguration",
        ),
    ),
    ProviderInfo(
        name="delegation",
        config_key="agent_delegation_provider_configuration",
        display_name="Agent Delegation",
        cli_name="delegation-sources",
        source_base=(
            "zav.agents_sdk.adapters.agent_delegation.agent_delegation_provider",
            "DelegableAgentsSource",
        ),
        config_model=(
            "zav.agents_sdk.adapters.agent_delegation.agent_delegation_provider",
            "AgentDelegationProviderConfiguration",
        ),
    ),
    ProviderInfo(
        name="processing",
        config_key="message_processing_provider_configuration",
        display_name="Message Processing",
        cli_name="processors",
        source_base=(
            "zav.agents_sdk.adapters.message_processing.message_processing_provider",
            "MessageProcessor",
        ),
        config_model=(
            "zav.agents_sdk.adapters.message_processing.message_processing_provider",
            "MessageProcessingProviderConfiguration",
        ),
    ),
    ProviderInfo(
        name="dispatch",
        config_key="dispatch_provider_configuration",
        display_name="Dispatch",
        cli_name="dispatch-rules",
        source_base=(
            "zav.agents_sdk.adapters.dispatch.dispatch_provider",
            "DispatchRule",
        ),
        config_model=(
            "zav.agents_sdk.adapters.dispatch.dispatch_provider",
            "DispatchProviderConfiguration",
        ),
    ),
    ProviderInfo(
        name="mcp",
        config_key="mcp_tools_provider_configuration",
        display_name="MCP Tools",
        cli_name="mcp-servers",
        source_base=None,
        config_model=(
            "zav.agents_sdk.adapters.mcp.tools_provider",
            "MCPToolsProviderConfiguration",
        ),
    ),
)

_PROVIDER_MAP: Dict[str, ProviderInfo] = {p.name: p for p in _PROVIDERS}

_SOURCE_CONFIG_KEY_OVERRIDES: Dict[Tuple[str, str], str] = {
    ("processing", "artifact_conductor"): "artifact_conductor_processor_configuration",
}


def get_providers() -> Dict[str, ProviderInfo]:
    return _PROVIDER_MAP


def get_provider_keys() -> Dict[str, str]:
    return {p.name: p.config_key for p in _PROVIDERS}


def get_provider_display_names() -> Dict[str, str]:
    return {p.name: p.display_name for p in _PROVIDERS}


def get_provider_cli_names() -> Dict[str, str]:
    return {p.name: p.cli_name for p in _PROVIDERS}


def get_provider_source_bases() -> Dict[str, Tuple[str, str]]:
    return {p.name: p.source_base for p in _PROVIDERS if p.source_base is not None}


def get_provider_config_models() -> Dict[str, Tuple[str, str]]:
    return {p.name: p.config_model for p in _PROVIDERS if p.config_model is not None}


@dataclass
class SourceMetadata:
    source_name: str
    cli_name: str
    config_key: str
    config_model: Optional[Type[Any]]
    factory: Any


class SourceRegistry:
    __cache: Dict[str, Dict[str, SourceMetadata]] = {}
    __adapters_loaded: bool = False

    @classmethod
    def __ensure_adapters_loaded(cls):
        if cls.__adapters_loaded:
            return
        cls.__adapters_loaded = True

        adapter_packages = [
            "zav.agents_sdk.adapters.tools",
            "zav.agents_sdk.adapters.skills",
            "zav.agents_sdk.adapters.memory",
            "zav.agents_sdk.adapters.context",
            "zav.agents_sdk.adapters.instructions",
            "zav.agents_sdk.adapters.agent_delegation",
            "zav.agents_sdk.adapters.message_processing",
            "zav.agents_sdk.adapters.dispatch",
        ]
        from importlib import import_module

        for pkg in adapter_packages:
            try:
                import_module(pkg)
            except ImportError:
                pass

    @classmethod
    def __discover_provider(cls, provider_name: str) -> Dict[str, SourceMetadata]:
        cls.__ensure_adapters_loaded()

        base_info = get_provider_source_bases().get(provider_name)
        if base_info is None:
            return {}

        from importlib import import_module

        mod = import_module(base_info[0])
        base_class = getattr(mod, base_info[1])

        # Deferred import: domain modules pull in heavy SDK internals — importing
        # at module level adds seconds to CLI startup for commands that don't
        # need them.
        from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

        factories = AgentDependencyRegistry.get_subclasses_of(base_class)
        result: Dict[str, SourceMetadata] = {}

        for factory in factories:
            sig = inspect.signature(factory.create)
            return_type = sig.return_annotation
            if return_type is inspect.Signature.empty:
                continue
            if not hasattr(return_type, "source_name"):
                continue

            source_name = return_type.source_name
            config_key = None
            config_model = None

            for param_name, param in sig.parameters.items():
                if param_name in ("cls", "self"):
                    continue
                ann = param.annotation
                if ann is inspect.Parameter.empty:
                    continue
                if isinstance(ann, type) and issubclass(ann, BaseModel):
                    if param_name.endswith("_configuration"):
                        config_key = param_name
                        config_model = ann
                        break

            if config_key is None:
                config_key = f"{source_name}_source_configuration"

            cli_name = source_name.replace("_", "-")

            result[source_name] = SourceMetadata(
                source_name=source_name,
                cli_name=cli_name,
                config_key=config_key,
                config_model=config_model,
                factory=factory,
            )

        return result

    @classmethod
    def get_sources(cls, provider_name: str) -> Dict[str, SourceMetadata]:
        if provider_name not in cls.__cache:
            cls.__cache[provider_name] = cls.__discover_provider(provider_name)
        return cls.__cache[provider_name]

    @classmethod
    def get_source(
        cls, provider_name: str, source_name: str
    ) -> Optional[SourceMetadata]:
        return cls.get_sources(provider_name).get(source_name)

    @classmethod
    def get_aliases(cls, provider_name: str) -> Dict[str, str]:
        sources = cls.get_sources(provider_name)
        return {meta.cli_name: meta.source_name for meta in sources.values()}

    @classmethod
    def clear_cache(cls):
        cls.__cache.clear()
        cls.__adapters_loaded = False


def get_source_config_key(provider: str, internal_name: str) -> str:
    meta = SourceRegistry.get_source(provider, internal_name)
    if meta is not None:
        return meta.config_key
    if (provider, internal_name) in _SOURCE_CONFIG_KEY_OVERRIDES:
        return _SOURCE_CONFIG_KEY_OVERRIDES[(provider, internal_name)]
    return f"{internal_name}_source_configuration"


def get_source_config_model(provider: str, internal_name: str) -> Optional[Type[Any]]:
    meta = SourceRegistry.get_source(provider, internal_name)
    if meta is not None:
        return meta.config_model
    return None


MODEL_VENDOR_MAP: Dict[str, str] = {
    "gpt-5.4": "openai",
    "gpt-5.4-mini": "openai",
    "gpt-5.4-nano": "openai",
    "gpt-4.1": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4.1-nano": "openai",
    "o3": "openai",
    "o4-mini": "openai",
    "claude-opus-4-7": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "claude-haiku-4-5": "anthropic",
}


def resolve_alias(provider: str, alias: str) -> str:
    sources = SourceRegistry.get_sources(provider)
    normalized = alias.replace("-", "_")
    if normalized in sources:
        return normalized
    for meta in sources.values():
        if meta.cli_name == alias:
            return meta.source_name
    available = ", ".join(sorted(m.cli_name for m in sources.values()))
    raise AliasError(
        f"Unknown source '{alias}' for provider '{provider}'. "
        f"Available: {available}"
    )


def reverse_alias(provider: str, internal_name: str) -> str:
    meta = SourceRegistry.get_source(provider, internal_name)
    if meta is not None:
        return meta.cli_name
    return internal_name.replace("_", "-")


def get_provider_key(provider: str) -> str:
    keys = get_provider_keys()
    if provider in keys:
        return keys[provider]
    raise AliasError(
        f"Unknown provider '{provider}'. "
        f"Available: {', '.join(sorted(keys.keys()))}"
    )


def list_provider_names() -> List[str]:
    return sorted(get_provider_keys().keys())


def list_source_aliases(provider: str) -> List[Tuple[str, str, str]]:
    sources = SourceRegistry.get_sources(provider)
    return [
        (meta.cli_name, meta.source_name, "")
        for meta in sorted(sources.values(), key=lambda m: m.cli_name)
    ]


def infer_vendor(model_name: str) -> Optional[str]:
    for prefix, vendor in MODEL_VENDOR_MAP.items():
        if model_name.startswith(prefix):
            return vendor
    if model_name.startswith("gpt-") or model_name.startswith("o"):
        return "openai"
    if model_name.startswith("claude-"):
        return "anthropic"
    return None


class AliasError(Exception):
    pass
