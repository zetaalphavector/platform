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

_UPDATE_MEMORY_TOOL_DESCRIPTION = """\
Update an existing memory entry. Use this when you learn that a \
previously saved fact is outdated, incorrect, or needs refinement. \
Provide the memory id (shown in the recalled memories list) and \
the full updated content. The new content completely replaces the old.\
You can call this tool multiple times, if more than one memory entry \
needs to be updated. \
"""


class MemoryEntry(BaseModel):
    id: Optional[str] = Field(
        None, description="Unique memory entry internal identifier."
    )
    content: str = Field(..., description="The memory content.")
    created_at: Optional[str] = Field(
        None, description="ISO date when the memory was created."
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata.")


class MemorySaveError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class MemoryUpdateError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class MemoryCapability(BaseModel):
    name: str = Field(..., description="Tool name.")
    description: str = Field(..., description="Description shown to the LLM.")
    parameters_spec: Dict[str, Any] = Field(
        ..., description="JSON Schema for the tool parameters."
    )


class MemoryStore(ABC):
    source_name: ClassVar[str]
    enabled: bool

    @abstractmethod
    async def load_context(self) -> List[MemoryEntry]:
        raise NotImplementedError

    @abstractmethod
    async def save(self, content: str) -> MemoryEntry:
        raise NotImplementedError

    async def update(self, entry_id: str, content: str) -> MemoryEntry:
        """Update an existing memory entry. Override in stores that support it."""
        raise MemoryUpdateError(
            f"Source '{self.source_name}' does not support updating memories."
        )

    def get_capabilities(self) -> List[MemoryCapability]:
        capabilities = [
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
        if type(self).update is not MemoryStore.update:
            capabilities.append(
                MemoryCapability(
                    name="update_memory",
                    description=_UPDATE_MEMORY_TOOL_DESCRIPTION,
                    parameters_spec={
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": (
                                    "The identifier of the memory entry to update, "
                                    "as shown in the recalled memories list."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": _SAVE_MEMORY_CONTENT_DESCRIPTION,
                            },
                        },
                        "required": ["id", "content"],
                    },
                ),
            )
        return capabilities

    def get_tools(self) -> List[Tool]:
        saved_entries: List[MemoryEntry] = []
        updated_entries: List[MemoryEntry] = []
        executables = self.__get_executables(saved_entries, updated_entries)
        tools: List[Tool] = []
        for cap in self.get_capabilities():
            executable = executables.get(cap.name)
            if executable is None:
                continue
            if cap.name == "save_memory":
                streaming_config = ToolStreamingConfig(
                    running_text="Saving fact to memory...",
                    completed_text="Saved fact to memory.",
                    response_transform=lambda r: (
                        {**r, "metadata": saved_entries[-1].metadata}
                        if r and saved_entries
                        else r
                    ),
                )
            else:
                streaming_config = ToolStreamingConfig(
                    running_text="Updating memory...",
                    completed_text="Updated memory.",
                    response_transform=lambda r: (
                        {**r, "metadata": updated_entries[-1].metadata}
                        if r and updated_entries
                        else r
                    ),
                )
            tools.append(
                Tool(
                    name=cap.name,
                    description=cap.description,
                    executable=executable,
                    parameters_spec=cap.parameters_spec,
                    streaming_config=streaming_config,
                )
            )
        return tools

    def __get_executables(
        self,
        saved_entries: List[MemoryEntry],
        updated_entries: List[MemoryEntry],
    ) -> Dict[str, Any]:
        async def save_or_error(content: str) -> Dict:
            try:
                entry = await self.save(content)
            except MemorySaveError as e:
                return {"error": str(e)}
            saved_entries.append(entry)
            return {"info": "Memory saved."}

        async def update_or_error(id: str, content: str) -> Dict:
            try:
                entry = await self.update(id, content)
            except MemoryUpdateError as e:
                return {"error": str(e)}
            updated_entries.append(entry)
            return {"info": f"Memory '{id}' updated."}

        executables: Dict[str, Any] = {"save_memory": save_or_error}
        if type(self).update is not MemoryStore.update:
            executables["update_memory"] = update_or_error
        return executables


class MemoryStoreGroup(DependencyGroup[MemoryStore]):
    __collects__ = MemoryStore
