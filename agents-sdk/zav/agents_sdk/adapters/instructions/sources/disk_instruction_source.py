from pathlib import Path
from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.instructions.instruction_source import InstructionSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class DiskInstructionSourceConfiguration(BaseModel):
    enabled: bool = False
    instructions_directories: List[str] = Field(
        default_factory=list,
        description="List of directories to scan for instruction markdown files.",
    )


class DiskInstructionSource(InstructionSource):

    source_name = "disk_instructions"

    def __init__(self, directories: List[str], enabled: bool):
        self.enabled = enabled
        self.__directories = directories
        self.__prompt: Optional[str] = None

    async def to_prompt(self) -> str:
        if self.__prompt is None:
            self.__prompt = self.__load_prompt()
        return self.__prompt

    def __load_prompt(self) -> str:
        parts: List[str] = []
        for directory in self.__directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning(f"Instructions directory not found: {directory}")
                continue

            for instruction_path in sorted(dir_path.iterdir()):
                if (
                    instruction_path.is_file()
                    and instruction_path.suffix.lower() == ".md"
                ):
                    content = self.__read_instruction(instruction_path)
                    if content:
                        parts.append(content)

        return "\n\n".join(parts)

    def __read_instruction(self, instruction_path: Path) -> str:
        try:
            return instruction_path.read_text().strip()
        except Exception as e:
            logger.warning(f"Failed to read instruction file {instruction_path}: {e}")
            return ""


class DiskInstructionSourceFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        disk_instruction_source_configuration: DiskInstructionSourceConfiguration = (
            DiskInstructionSourceConfiguration()
        ),
    ) -> DiskInstructionSource:
        return DiskInstructionSource(
            directories=disk_instruction_source_configuration.instructions_directories,
            enabled=disk_instruction_source_configuration.enabled,
        )
