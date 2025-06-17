import enum
from typing import Any, Dict, List, Literal, Optional, Union

from zav.pydantic_compat import PYDANTIC_V2, BaseModel, root_validator


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


class FunctionCallResponse(BaseModel):
    name: str
    result: Optional[str] = None

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

    @root_validator()
    @classmethod
    def at_most_one(cls, values):
        # ensure only one of document_context and custom_context
        if PYDANTIC_V2:
            doc = values.document_context
            custom = values.custom_context
        else:
            doc = values.get("document_context")
            custom = values.get("custom_context")
        if doc and custom:
            raise ValueError(
                "At most one of document_context and custom_context can be set"
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


class ContentPartTool(BaseModel):
    name: str
    params: Optional[Dict] = None
    response: Optional[Dict] = None


class ContentPartTable(BaseModel):
    format: Literal["row", "columnar"] = "row"
    rows: Optional[List[Dict[str, str]]] = None
    columns: Optional[Dict[str, List[str]]] = None
    headers: Optional[List[str]] = None

    @root_validator()
    @classmethod
    def one_of(cls, v):
        """Verify exactly one of 'rows' or 'columns' is provided."""
        # accept v as dict (v1) or model instance (v2)
        if PYDANTIC_V2:
            rows = v.rows
            columns = v.columns
        else:
            rows = v.get("rows")
            columns = v.get("columns")
        # At least one must be set
        if rows is None and columns is None:
            raise ValueError(
                "At least one of the fields 'rows' and 'columns' must have a value"
            )
        # Only one may be set
        if rows is not None and columns is not None:
            raise ValueError(
                "Only one of the fields 'rows' and 'columns' must have a value."
            )
        return v


class ContentPart(BaseModel):
    type: Literal["context", "text", "tool", "table"]
    context: Optional[ConversationContext] = None
    tool: Optional[ContentPartTool] = None
    text: Optional[str] = None
    table: Optional[ContentPartTable] = None


class ChatMessage(BaseModel):
    sender: ChatMessageSender
    content: str
    content_parts: Optional[List[ContentPart]] = None
    image_uri: Optional[str] = None
    function_call_request: Optional[FunctionCallRequest] = None
    function_call_response: Optional[FunctionCallResponse] = None
    evidences: Optional[List[ChatMessageEvidence]] = None
    function_specs: Optional[FunctionSpec] = None

    class Config:
        orm_mode = True
