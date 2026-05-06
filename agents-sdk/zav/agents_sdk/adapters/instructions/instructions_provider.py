from typing import Any, Dict, List, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.instructions.instruction_source import (
    InstructionSource,
    InstructionSourceGroup,
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


class InstructionsProvider:

    def __init__(
        self,
        sources: List[InstructionSource],
        enabled: bool,
        include_sources: Optional[Set[str]] = None,
        exclude_sources: Optional[Set[str]] = None,
    ):
        self.__sources = sources
        self.__enabled = enabled
        self.__include_sources = include_sources
        self.__exclude_sources = exclude_sources
        self.__cached_active: Optional[List[InstructionSource]] = None

    def describe_loaded(self) -> Dict[str, Any]:
        return {
            "enabled": self.__enabled,
            "sources": [s.source_name for s in self.__active_sources()],
        }

    async def to_prompt(self) -> str:
        if not self.__enabled:
            return ""
        parts: List[str] = []
        for source in self.__active_sources():
            try:
                prompt = await source.to_prompt()
                if prompt:
                    parts.append(prompt)
            except Exception as e:
                logger.error(
                    f"Failed to get instructions from {source.source_name}: {e}"
                )
        return _PROMPT_SEPARATOR.join(parts)

    def __active_sources(self) -> List[InstructionSource]:
        if self.__cached_active is not None:
            return self.__cached_active
        self.__cached_active = [
            source
            for source in self.__sources
            if is_source_active(
                source.source_name,
                source.enabled,
                self.__include_sources,
                self.__exclude_sources,
            )
        ]
        return self.__cached_active


class InstructionsProviderFactory(AgentDependencyFactory):

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
        )
