from pathlib import Path
from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.system_prompt.system_prompt_source import (
    SystemPromptSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class DiskSystemPromptSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable the system prompt files.")
    files: List[str] = Field(
        default_factory=list,
        description="Files joined in this order.",
    )


class DiskSystemPromptSource(SystemPromptSource):

    source_name = "disk_system_prompt"
    source_description = "System prompt read from files on disk."
    source_gate = "local_files"

    def __init__(self, enabled: bool, files: List[str]):
        self.enabled = enabled
        self.__files = files
        self.__prompt: Optional[str] = None

    async def to_prompt(self) -> str:
        if self.__prompt is None:
            self.__prompt = "\n\n".join(
                part for part in map(self.__read, self.__files) if part
            )
        return self.__prompt

    @staticmethod
    def __read(file: str) -> str:
        try:
            return Path(file).read_text().strip()
        except Exception as e:
            logger.warning(f"Failed to read system prompt file {file}: {e}")
            return ""


class DiskSystemPromptSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        disk_system_prompt_source_configuration: DiskSystemPromptSourceConfiguration = (
            DiskSystemPromptSourceConfiguration()
        ),
    ) -> DiskSystemPromptSource:
        return DiskSystemPromptSource(
            enabled=disk_system_prompt_source_configuration.enabled,
            files=disk_system_prompt_source_configuration.files,
        )
