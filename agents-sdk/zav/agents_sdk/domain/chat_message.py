import enum
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import parse_qs, urlparse

from zav.pydantic_compat import PYDANTIC_V2, BaseModel, ConfigDict, root_validator

from zav.agents_sdk.domain.tools import ToolDisplay


class ChatMessageSender(str, enum.Enum):
    USER = "user"
    BOT = "bot"


class ChatMessageEvidence(BaseModel):
    # `document_hit_url` is a legacy field name: it holds an evidence URL that
    # may be either an internal `/documents/...` retrieval URL or an external
    # URL (e.g. for federated search results). Consumers that need to derive an
    # internal document id should go through
    # `extract_internal_document_id_from_evidence_url`.
    document_hit_url: str
    text_extract: Optional[str] = None
    anchor_text: Optional[str] = None


def extract_internal_document_id_from_evidence_url(
    evidence_url: str,
) -> Optional[str]:
    parsed_url = urlparse(evidence_url)
    if parsed_url.scheme or parsed_url.netloc:
        return None
    if not parsed_url.path.startswith("/documents/"):
        return None

    property_values = parse_qs(parsed_url.query).get("property_values")
    if not property_values or not property_values[0]:
        return None

    return property_values[0].split("_")[0]


class FunctionCallRequest(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True


class FunctionCallResponse(BaseModel):
    name: str
    result: Optional[str] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:

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


class TagContext(BaseModel):
    tag_ids: List[str]

    def is_empty(self) -> bool:
        return not self.tag_ids


class FilterContext(BaseModel):
    filters: Dict  # Boolean tree format (FiltersConfiguration-compatible)


class UserDocumentContext(BaseModel):
    enabled: bool = True


class ConversationContext(BaseModel):
    document_context: Optional[DocumentContext] = None
    custom_context: Optional[CustomContext] = None
    tag_context: Optional[TagContext] = None
    filter_context: Optional[FilterContext] = None
    user_document_context: Optional[UserDocumentContext] = None

    def is_empty(self):
        return (
            (not self.document_context or not self.document_context.document_ids)
            and (not self.custom_context or not self.custom_context.items)
            and (not self.tag_context or self.tag_context.is_empty())
            and not self.filter_context
            and not self.user_document_context
        )


class ContentPartTool(BaseModel):
    tool_call_id: str
    name: str
    params: Optional[Dict] = None
    response: Optional[Dict] = None
    display_text: Optional[str] = None
    status: Optional[Literal["running", "completed", "error"]] = None
    display: Optional[ToolDisplay] = None


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


class ContentPartImage(BaseModel):
    # A data URI (`data:<mime>;base64,<data>`), as `ChatMessage.image_uri`.
    image_uri: str


class ContentPart(BaseModel):
    # "user_message": a message the user sent while the turn ran, in `text`,
    # where the agent read it; `message_id` is the id it was sent with.
    type: Literal["context", "text", "tool", "table", "image", "user_message"]
    context: Optional[ConversationContext] = None
    tool: Optional[ContentPartTool] = None
    text: Optional[str] = None
    table: Optional[ContentPartTable] = None
    image: Optional[ContentPartImage] = None
    message_id: Optional[str] = None


class ChatMessage(BaseModel):
    sender: ChatMessageSender
    content: str
    message_id: Optional[str] = None
    session_id: Optional[str] = None
    content_parts: Optional[List[ContentPart]] = None
    image_uri: Optional[str] = None
    function_call_request: Optional[FunctionCallRequest] = None
    function_call_response: Optional[FunctionCallResponse] = None
    evidences: Optional[List[ChatMessageEvidence]] = None
    function_specs: Optional[FunctionSpec] = None
    reasoning_summary: Optional[str] = None
    # The LLM configuration a bot turn ran on, stamped by the chat handlers.
    llm_configuration_name: Optional[str] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True

    @classmethod
    def as_user(cls, content: str) -> "ChatMessage":
        return cls(sender=ChatMessageSender.USER, content=content)

    @classmethod
    def as_bot(cls, content: str) -> "ChatMessage":
        return cls(sender=ChatMessageSender.BOT, content=content)

    def all_image_uris(self) -> List[str]:
        """Every image on the message: the single `image_uri`, then image parts."""
        part_images = [
            part.image.image_uri
            for part in self.content_parts or []
            if part.type == "image" and part.image
        ]
        return ([self.image_uri] if self.image_uri else []) + part_images
