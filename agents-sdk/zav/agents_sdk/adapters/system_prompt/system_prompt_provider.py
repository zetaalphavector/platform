from typing import Any, Dict, List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import active_sources
from zav.agents_sdk.adapters.system_prompt.system_prompt_source import (
    SystemPromptSource,
    SystemPromptSourceGroup,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_PROMPT_SEPARATOR = "\n\n"


class SystemPromptProviderConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable system prompt sources.")
    include_sources: Optional[List[str]] = Field(
        None,
        description=(
            "Allowlist of system prompt source names, in the order their text "
            "is joined. Unset: every enabled source, in registration order."
        ),
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of system prompt source names to exclude."
    )


class SystemPromptProvider:
    def __init__(
        self,
        sources: List[SystemPromptSource],
        enabled: bool,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ):
        self.__enabled = enabled
        active = active_sources(
            sources,
            enabled,
            set(include_sources) if include_sources else None,
            set(exclude_sources) if exclude_sources else None,
        )
        if include_sources:
            active.sort(key=lambda source: include_sources.index(source.source_name))
        self.__active = active

    def describe_loaded(self) -> Dict[str, Any]:
        return {
            "enabled": self.__enabled,
            "sources": [source.source_name for source in self.__active],
        }

    async def to_prompt(self) -> str:
        parts: List[str] = []
        for source in self.__active:
            try:
                prompt = await source.to_prompt()
            except Exception as e:
                logger.error(
                    f"Failed to get the system prompt from {source.source_name}: {e}"
                )
                continue
            if prompt:
                parts.append(prompt)
        return _PROMPT_SEPARATOR.join(parts)


class SystemPromptProviderFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls,
        system_prompt_source_group: SystemPromptSourceGroup = SystemPromptSourceGroup(
            items=[]
        ),
        system_prompt_provider_configuration: SystemPromptProviderConfiguration = (
            SystemPromptProviderConfiguration()
        ),
    ) -> SystemPromptProvider:
        return SystemPromptProvider(
            sources=system_prompt_source_group.items,
            enabled=system_prompt_provider_configuration.enabled,
            include_sources=system_prompt_provider_configuration.include_sources,
            exclude_sources=system_prompt_provider_configuration.exclude_sources,
        )
