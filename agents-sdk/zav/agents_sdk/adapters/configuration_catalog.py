import inspect
from typing import Any, Dict, List, Optional, Tuple, get_args

from zav.pydantic_compat import PYDANTIC_V2, BaseModel

from zav.agents_sdk.adapters.agent_delegation.agent_delegation_provider import (
    AgentDelegationProvider,
)
from zav.agents_sdk.adapters.agent_delegation.delegable_agents_source import (
    DelegableAgentsSource,
)
from zav.agents_sdk.adapters.context.context_provider import ContextProvider
from zav.agents_sdk.adapters.context.context_source import ContextSource
from zav.agents_sdk.adapters.dispatch.dispatch_provider import DispatchProvider
from zav.agents_sdk.adapters.dispatch.dispatch_rule import DispatchRule
from zav.agents_sdk.adapters.instructions.instruction_source import InstructionSource
from zav.agents_sdk.adapters.instructions.instructions_provider import (
    InstructionsProvider,
)
from zav.agents_sdk.adapters.interjections.interjection_source import (
    InterjectionSource,
)
from zav.agents_sdk.adapters.interjections.interjections_provider import (
    InterjectionsProvider,
)
from zav.agents_sdk.adapters.llm_models.queued_messages import QueuedMessages
from zav.agents_sdk.adapters.mcp.tools_provider import MCPToolsProvider
from zav.agents_sdk.adapters.memory.memory_provider import MemoryProvider
from zav.agents_sdk.adapters.memory.memory_store import MemoryStore
from zav.agents_sdk.adapters.message_processing.message_processing_provider import (
    MessageProcessingProvider,
)
from zav.agents_sdk.adapters.message_processing.message_processor import (
    MessageProcessor,
)
from zav.agents_sdk.adapters.queued_messages.queued_messages_source import (
    QueuedMessagesSource,
)
from zav.agents_sdk.adapters.repl.repl_provider import ReplProvider
from zav.agents_sdk.adapters.repl.repl_sandbox import ReplSandbox
from zav.agents_sdk.adapters.skills.skills_provider import SkillsProvider
from zav.agents_sdk.adapters.skills.skills_source import SkillsSource
from zav.agents_sdk.adapters.tool_policy import ToolPolicyProvider, ToolPolicySource
from zav.agents_sdk.adapters.tool_search.tool_search_provider import (
    ToolSearchProvider,
)
from zav.agents_sdk.adapters.tool_search.tool_search_source import ToolSearchSource
from zav.agents_sdk.adapters.tools.tools_provider import ToolsProvider
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import (
    AgentDependencyRegistryProtocol,
    DependencyGroup,
)
from zav.agents_sdk.domain.utils import check_is_base_model, check_is_optional

if PYDANTIC_V2:
    from pydantic.json_schema import GenerateJsonSchema
    from pydantic_core import PydanticOmit

    class _SettingsSchema(GenerateJsonSchema):
        # Fields only code can set (callables) have no JSON form; leave them out.
        def handle_invalid_for_json_schema(self, schema: Any, error_info: str) -> Any:
            raise PydanticOmit


def _json_schema(model: Any) -> Dict[str, Any]:
    if PYDANTIC_V2:
        return model.model_json_schema(schema_generator=_SettingsSchema)
    return model.schema()


_PROVIDERS: List[Tuple[type, str]] = [
    (ToolsProvider, "tools"),
    (MCPToolsProvider, "mcp"),
    (SkillsProvider, "skills"),
    (AgentDelegationProvider, "delegation"),
    (MemoryProvider, "memory"),
    (ContextProvider, "context"),
    (InstructionsProvider, "instructions"),
    (InterjectionsProvider, "interjections"),
    (ReplProvider, "repl"),
    (MessageProcessingProvider, "message_processing"),
    (DispatchProvider, "dispatch"),
    (ToolSearchProvider, "tool_search"),
    (ToolPolicyProvider, "tool_policy"),
    (QueuedMessages, "queued_messages"),
]
_SOURCES: List[Tuple[type, str]] = [
    (ToolsSource, "tools"),
    (SkillsSource, "skills"),
    (DelegableAgentsSource, "delegation"),
    (MemoryStore, "memory"),
    (ContextSource, "context"),
    (InstructionSource, "instructions"),
    (InterjectionSource, "interjections"),
    (ReplSandbox, "repl"),
    (MessageProcessor, "message_processing"),
    (DispatchRule, "dispatch"),
    (ToolSearchSource, "tool_search"),
    (ToolPolicySource, "tool_policy"),
    (QueuedMessagesSource, "queued_messages"),
]

# Capabilities behind a tenant gate: the agent editor hides each group until the
# tenant enables its gate.
_GATED_PROVIDERS = {"interjections": "interjections", "repl": "python_repl"}
_GATED_BLOCKS = {
    "file_attempt_store_configuration": "attempt_lineage",
    "in_memory_attempt_store_configuration": "attempt_lineage",
    "exact_match_scorer_configuration": "attempt_lineage",
    "llm_judge_scorer_configuration": "attempt_lineage",
    "refiner_configuration": "attempt_lineage",
    "disk_rubric_source_configuration": "attempt_lineage",
    "in_memory_rubric_source_configuration": "attempt_lineage",
}
# Blocks that belong to a provider without being its own or a source's.
_BLOCK_GROUPS = {
    "llm_selection_configuration": "policies",
    "image_llm_selection_configuration": "policies",
    "citation_configuration": "policies",
    "context_window_configuration": "policies",
    "sub_agent_limits_configuration": "delegation",
    "recent_memory_context_strategy_configuration": "memory",
    "my_documents_argument_resolver_configuration": "mcp",
}
# Providers whose blocks show with the policies rather than as a capability.
_PROVIDER_GROUPS = {"tool_search": "policies", "tool_policy": "policies"}
# Settings only a local project uses.
_INTERNAL_BLOCKS = {"local_file_mcp_oauth_integration_store_configuration"}


# A block several dependencies read belongs to its strongest owner: a provider
# reads its own settings, a source (a rule over the tools settings) may too.
_RANK = {"other": 0, "source": 1, "provider": 2}

# kind, provider, source_name, source_description, source_gate
_Owner = Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]


class ConfigurationSchema(BaseModel):
    # The key in an agent's configuration, e.g. `index_tools_source_configuration`.
    key: str
    # `provider` (a provider's own settings), `source` (one of its toolsets) or
    # `other`.
    kind: str
    provider: Optional[str] = None
    source_name: Optional[str] = None
    source_description: Optional[str] = None
    # Where it shows: a provider, `policies`, or None for blocks no documented
    # capability owns.
    group: Optional[str] = None
    # The tenant gate it sits behind, hidden until a tenant enables it.
    gate: Optional[str] = None
    json_schema: Dict[str, Any]


def _owner(created: type) -> _Owner:
    for base, provider in _PROVIDERS:
        if issubclass(created, base):
            return "provider", provider, None, None, None
    for base, provider in _SOURCES:
        if issubclass(created, base):
            return (
                "source",
                provider,
                getattr(created, "source_name", None),
                getattr(created, "source_description", None),
                getattr(created, "source_gate", None),
            )
    return "other", None, None, None, None


def _record(
    schemas: Dict[str, ConfigurationSchema],
    name: str,
    model: type,
    owner: _Owner,
) -> None:
    kind, provider, source_name, source_description, source_gate = owner
    existing = schemas.get(name)
    if (
        not name.endswith("_configuration")
        or name in _INTERNAL_BLOCKS
        or (existing is not None and _RANK[existing.kind] >= _RANK[kind])
    ):
        return
    group = (
        _PROVIDER_GROUPS.get(provider or "", provider)
        if kind != "other"
        else _BLOCK_GROUPS.get(name)
    )
    schemas[name] = ConfigurationSchema(
        key=name,
        kind=kind,
        provider=provider,
        source_name=source_name,
        source_description=source_description,
        group=group,
        gate=(
            _GATED_PROVIDERS.get(provider or "")
            or source_gate
            or _GATED_BLOCKS.get(name)
        ),
        json_schema=_json_schema(model),
    )


class _Walk:
    """Follows dependencies the way the resolver does: a parameter is a
    configuration block, a registered dependency, or a group of every
    registered dependency of a type."""

    def __init__(self, registry: AgentDependencyRegistryProtocol) -> None:
        self.__registry = registry
        self.__entries = registry.entries()
        self.__seen: set = set()
        self.schemas: Dict[str, ConfigurationSchema] = {}

    def parameters(
        self,
        parameters: Any,
        owner: _Owner,
    ) -> None:
        for name, param in parameters.items():
            annotation = param.annotation
            if check_is_optional(annotation):
                annotation = next(
                    a for a in get_args(annotation) if a is not type(None)
                )
            if not inspect.isclass(annotation):
                continue
            if check_is_base_model(annotation):
                _record(self.schemas, name, annotation, owner)
            elif issubclass(annotation, DependencyGroup) and hasattr(
                annotation, "__collects__"
            ):
                for created, member in self.__entries:
                    if inspect.isclass(created) and issubclass(
                        created, annotation.__collects__
                    ):
                        self.factory(created, member)
            else:
                registered = self.__registry.get(annotation)
                if registered is not None:
                    self.factory(annotation, registered)

    def factory(self, created: type, factory: Any) -> None:
        if created in self.__seen:
            return
        self.__seen.add(created)
        self.parameters(inspect.signature(factory.create).parameters, _owner(created))


def configuration_schemas(
    registry: AgentDependencyRegistryProtocol,
    agent_class: Optional[type] = None,
) -> List[ConfigurationSchema]:
    """The configuration blocks an agent class can read, each with the provider
    or source it configures; every registered block without a class. A block
    several dependencies read is listed once, under a provider or source owner
    when it has one."""
    walk = _Walk(registry)
    if agent_class is None:
        for created, factory in registry.entries():
            if inspect.isclass(created):
                walk.factory(created, factory)
    else:
        walk.parameters(
            inspect.signature(agent_class).parameters, ("other", None, None, None, None)
        )
    return sorted(walk.schemas.values(), key=lambda s: s.key)
