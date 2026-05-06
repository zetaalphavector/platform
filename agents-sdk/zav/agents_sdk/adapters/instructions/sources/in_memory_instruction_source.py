from typing import List

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.instructions.instruction_source import InstructionSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class InMemoryInstructionDefinition(BaseModel):
    header: str = Field(..., description="Section header for this instruction block.")
    body: str = Field(..., description="The instruction text to inject.")


class InMemoryInstructionSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable in-memory instruction source.")
    instructions: List[InMemoryInstructionDefinition] = Field(
        default_factory=list,
        description="List of instruction blocks to inject.",
    )


class InMemoryInstructionSource(InstructionSource):

    source_name = "in_memory_instructions"

    def __init__(
        self,
        enabled: bool,
        instructions: List[InMemoryInstructionDefinition],
    ):
        self.enabled = enabled
        self.__instructions = instructions

    async def to_prompt(self) -> str:
        parts = []
        for instruction in self.__instructions:
            if instruction.body:
                parts.append(f"## {instruction.header}\n\n{instruction.body}")
        return "\n\n".join(parts)


class InMemoryInstructionSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        in_memory_instruction_source_configuration: InMemoryInstructionSourceConfiguration = InMemoryInstructionSourceConfiguration(),  # noqa: E501
    ) -> InMemoryInstructionSource:
        return InMemoryInstructionSource(
            instructions=in_memory_instruction_source_configuration.instructions,
            enabled=in_memory_instruction_source_configuration.enabled,
        )
