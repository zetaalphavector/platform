from typing import Any, Dict, List, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import active_sources, pick_writable
from zav.agents_sdk.adapters.instructions.instruction_source import (
    InstructionSource,
    InstructionSourceGroup,
    WritableInstructionSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_PROMPT_SEPARATOR = "\n\n"


class InstructionsProviderConfiguration(BaseModel):
    enabled: bool = Field(True, description="Enable instruction sources.")
    include_sources: Optional[List[str]] = Field(
        None, description="Allowlist of instruction source names to include."
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of instruction source names to exclude."
    )
    writable_source: Optional[str] = Field(
        None,
        description=(
            "Source that receives prompt notes written on the agent's behalf "
            "(refinement). Must support writing; default: the first active "
            "source that does."
        ),
    )


class InstructionsProvider:

    def __init__(
        self,
        sources: List[InstructionSource],
        enabled: bool,
        include_sources: Optional[Set[str]] = None,
        exclude_sources: Optional[Set[str]] = None,
        writable_source: Optional[str] = None,
    ):
        self.__enabled = enabled
        self.__active = active_sources(
            sources, enabled, include_sources, exclude_sources
        )
        self.__writable = pick_writable(
            [s for s in self.__active if isinstance(s, WritableInstructionSource)],
            writable_source,
        )

    @property
    def active_sources(self) -> List[InstructionSource]:
        return self.__active

    @property
    def writable_source(self) -> Optional[WritableInstructionSource]:
        """Where prompt notes are written on the agent's behalf. None when no
        active source can be written to."""
        return self.__writable

    def describe_loaded(self) -> Dict[str, Any]:
        return {
            "enabled": self.__enabled,
            "sources": [s.source_name for s in self.__active],
        }

    async def to_prompt(self) -> str:
        if not self.__enabled:
            return ""
        parts: List[str] = []
        for source in self.__active:
            try:
                prompt = await source.to_prompt()
                if prompt:
                    parts.append(prompt)
            except Exception as e:
                logger.error(
                    f"Failed to get instructions from {source.source_name}: {e}"
                )
        return _PROMPT_SEPARATOR.join(parts)


class InstructionsProviderFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls,
        instruction_source_group: InstructionSourceGroup = InstructionSourceGroup(
            items=[]
        ),
        instructions_provider_configuration: InstructionsProviderConfiguration = (
            InstructionsProviderConfiguration()
        ),
    ) -> InstructionsProvider:
        return InstructionsProvider(
            sources=instruction_source_group.items,
            enabled=instructions_provider_configuration.enabled,
            include_sources=(
                set(instructions_provider_configuration.include_sources)
                if instructions_provider_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(instructions_provider_configuration.exclude_sources)
                if instructions_provider_configuration.exclude_sources
                else None
            ),
            writable_source=instructions_provider_configuration.writable_source,
        )
