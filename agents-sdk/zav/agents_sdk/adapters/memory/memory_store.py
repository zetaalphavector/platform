from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_dependency import DependencyGroup
from zav.agents_sdk.domain.tools import Tool, ToolStreamingConfig

_SAVE_MEMORY_TOOL_DESCRIPTION = """\
Save a fact to memory so your future self can recall it in later \
conversations. Memories persist across sessions.

What to save:
- User identity and profile (name, role, team, expertise level)
- User preferences (communication style, verbosity, language)
- User corrections ("we use pnpm, not npm", "don't use relative imports")
- Project conventions and architectural decisions discovered during work
- Recurring patterns, gotchas, or domain-specific knowledge
- Key decisions and their rationale

What NOT to save:
- Transient task progress or intermediate status updates
- Information already obvious from the codebase or documentation
- Raw data, large code snippets, or full file contents
- Facts you are not confident about

Write each memory as a clear, self-contained statement that your \
future self can understand without the original conversation context.\
"""

_SAVE_MEMORY_CONTENT_DESCRIPTION = """\
A clear, self-contained statement of the fact or convention to \
remember. Should make sense without the original conversation context.\
"""


class MemoryEntry(BaseModel):
    content: str = Field(..., description="The memory content.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata.")


class MemoryCapability(BaseModel):
    name: str = Field(..., description="Tool name.")
    description: str = Field(..., description="Description shown to the LLM.")
    parameters_spec: Dict[str, Any] = Field(
        ..., description="JSON Schema for the tool parameters."
    )


class MemoryStore(ABC):
    source_name: ClassVar[str]

    @abstractmethod
    async def load_context(self) -> List[MemoryEntry]:
        raise NotImplementedError

    @abstractmethod
    async def save(self, content: str) -> str:
        raise NotImplementedError

    def get_capabilities(self) -> List[MemoryCapability]:
        return [
            MemoryCapability(
                name="save_memory",
                description=_SAVE_MEMORY_TOOL_DESCRIPTION,
                parameters_spec={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": _SAVE_MEMORY_CONTENT_DESCRIPTION,
                        },
                    },
                    "required": ["content"],
                },
            ),
        ]

    def get_tools(self) -> List[Tool]:
        executables = self.__get_executables()
        tools: List[Tool] = []
        for cap in self.get_capabilities():
            executable = executables.get(cap.name)
            if executable is None:
                continue
            tools.append(
                Tool(
                    name=cap.name,
                    description=cap.description,
                    executable=executable,
                    parameters_spec=cap.parameters_spec,
                    streaming_config=ToolStreamingConfig(
                        running_text="Saving fact to memory...",
                        completed_text="Saved fact to memory",
                    ),
                )
            )
        return tools

    def __get_executables(self) -> Dict[str, Any]:
        return {"save_memory": self.save}


class MemoryStoreGroup(DependencyGroup[MemoryStore]):
    __collects__ = MemoryStore
