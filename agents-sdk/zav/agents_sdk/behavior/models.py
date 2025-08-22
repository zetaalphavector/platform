from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from zav.pydantic_compat import BaseModel, Field, validator

from zav.agents_sdk.domain.chat_message import CustomContext, DocumentContext


class BaseExpectation(BaseModel):
    """Common base class so we can Union multiple expectations."""

    type: str

    class Config:
        extra = "allow"


class TextIncludesExpectation(BaseExpectation):
    type: Literal["text_includes"] = "text_includes"
    value: str


class JsonIncludesExpectation(BaseExpectation):
    type: Literal["json_includes"] = "json_includes"
    items: Dict[str, Any]


class ToolCallExpectation(BaseExpectation):
    type: Literal["tool_call"] = "tool_call"
    name: str
    params_partial: Optional[Dict[str, Any]] = None


class CitationExpectation(BaseExpectation):
    type: Literal["citation"] = "citation"
    min_citations: Optional[int] = None
    cited_docs: Optional[List[str]] = None


Expectation = Union[
    TextIncludesExpectation,
    JsonIncludesExpectation,
    ToolCallExpectation,
    CitationExpectation,
]


class MessageSpec(BaseModel):
    """A conversation message provided in the YAML spec."""

    role: Literal["user", "bot"]
    content: str = Field(..., description="Raw message content")


class ConversationContextSpec(BaseModel):
    """A conversation context provided in the YAML spec."""

    document_context: Optional[DocumentContext] = None
    custom_context: Optional[CustomContext] = None


class TestSpecification(BaseModel):
    """Root model representing a .yaml test spec."""

    id: str
    description: Optional[str] = None
    agent_identifier: str
    messages: List[MessageSpec]
    conversation_context: Optional[ConversationContextSpec] = None
    expectations: List[Expectation] = Field(default_factory=list)
    bot_params: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"

    @validator("messages", pre=True)
    @classmethod
    def _coerce_messages(cls, v):
        """Allow YAML to omit explicit role list (future flexibility)."""
        if not isinstance(v, list):
            raise TypeError("messages must be a list")
        return v


def load_spec(path: Union[str, Path]) -> TestSpecification:
    """Load and parse a YAML specification file into a `TestSpecification`."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Spec file {path} is empty")

    return TestSpecification.validate(data)


__all__: List[str] = [
    "MessageSpec",
    "BaseExpectation",
    "TextIncludesExpectation",
    "JsonIncludesExpectation",
    "ToolCallExpectation",
    "CitationExpectation",
    "Expectation",
    "TestSpecification",
    "load_spec",
]
