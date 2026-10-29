from abc import ABC
from fnmatch import fnmatchcase
from typing import ClassVar, Iterable, List, Optional, Sequence

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.llm_models.chat_completion_types import ChatCompletion
from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.tools import Tool, ToolOptions


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    """Whether a tool name matches one of the names or patterns (`server__*`)."""
    return any(fnmatchcase(name, pattern) for pattern in patterns)


class ToolPolicySource(ABC):
    """A pluggable tool rule, switched on and off per tenant and agent with the
    tool policy's include_sources/exclude_sources like any source. Who the
    caller is reaches a rule through its factory."""

    source_name: ClassVar[str]
    # One line for admins switching rule sources on or off.
    source_description: ClassVar[Optional[str]] = None
    # Run order, from general to specific (agent < tenant and agent policies <
    # user): each source works on what the ones before it left.
    specificity: ClassVar[int] = 0
    enabled: bool

    async def apply(
        self, tools: List[Tool], transcript: Optional[List[ChatCompletion]]
    ) -> List[Tool]:
        """The tools kept, each with the options and callers this source sets,
        given the transcript so far. ``transcript`` is None outside a
        conversation (an MCP client listing the tools), so rules that only
        shape what a model sees step aside."""
        return tools


class ToolPolicySourceGroup(DependencyGroup[ToolPolicySource]):
    __collects__ = ToolPolicySource


class ToolPolicyConfiguration(BaseModel):
    include_sources: Optional[List[str]] = Field(
        None,
        description=(
            "Allowlist of tool rule sources. A rule only removes tools or adds "
            "confirmation, so switching one on never widens the agent."
        ),
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of tool rule sources."
    )


def active_rule_sources(
    sources: Sequence[ToolPolicySource],
    configuration: ToolPolicyConfiguration,
) -> List[ToolPolicySource]:
    """The sources that run, in run order."""
    include = set(configuration.include_sources or []) or None
    exclude = set(configuration.exclude_sources or []) or None
    return sorted(
        (
            source
            for source in sources
            if is_source_active(source.source_name, source.enabled, include, exclude)
        ),
        key=lambda source: source.specificity,
    )


def ask_first(options: ToolOptions, source_name: str, required: bool) -> ToolOptions:
    """A required ask always holds and no later source waives it. A default ask
    only fills in where no source has an opinion yet."""
    if options.confirmation == "require":
        return options
    if required:
        return options.copy(
            update={"confirmation": "require", "confirmation_source": source_name}
        )
    if options.confirmation != "auto":
        return options
    return options.copy(
        update={"confirmation": "ask", "confirmation_source": source_name}
    )


def waive_asking(
    options: ToolOptions, source_name: str, default: bool = False
) -> ToolOptions:
    """No required ask is waived. A default only fills in where no source has
    an opinion yet."""
    if options.confirmation == "require" or (
        default and options.confirmation != "auto"
    ):
        return options
    return options.copy(
        update={"confirmation": "never", "confirmation_source": source_name}
    )


def load_on_demand(options: ToolOptions, on_demand: bool) -> ToolOptions:
    return options.copy(update={"on_demand": on_demand})
