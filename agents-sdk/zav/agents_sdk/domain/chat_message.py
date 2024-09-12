import enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, root_validator


class ChatMessageSender(str, enum.Enum):
    USER = "user"
    BOT = "bot"


class ChatMessageEvidence(BaseModel):
    document_hit_url: str
    text_extract: Optional[str] = None
    anchor_text: Optional[str] = None


class FunctionCallRequest(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


class FunctionSpec(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class DocumentContext(BaseModel):
    document_ids: List[str]
    retrieval_unit: str


class CustomContextItem(BaseModel):
    document_id: str
    content: Union[str, Dict[str, Any]]

    def get_custom_hit_url(self):
        return f"custom://{self.document_id}"


class CustomContext(BaseModel):
    items: List[CustomContextItem]


class ConversationContext(BaseModel):
    document_context: Optional[DocumentContext] = None
    custom_context: Optional[CustomContext] = None

    @root_validator
    @classmethod
    def at_most_one(cls, values):
        document_context = values.get("document_context")
        custom_resources = values.get("custom_resources")
        if document_context and custom_resources:
            raise ValueError(
                "At most one of document_context and custom_resources can be set"
            )
        return values

    def is_empty(self):
        return (
            not self.document_context
            or (self.document_context and not self.document_context.document_ids)
        ) and (
            not self.custom_context
            or (self.custom_context and not self.custom_context.items)
        )


class ContentPart(BaseModel):
    type: Literal["context", "text"]
    context: Optional[ConversationContext] = None
    text: Optional[str] = None


class ChatMessage(BaseModel):
    sender: ChatMessageSender
    content: str
    content_parts: Optional[List[ContentPart]] = None
    image_uri: Optional[str] = None
    function_call_request: Optional[FunctionCallRequest] = None
    evidences: Optional[List[ChatMessageEvidence]] = None
    function_specs: Optional[FunctionSpec] = None

    class Config:
        orm_mode = True
