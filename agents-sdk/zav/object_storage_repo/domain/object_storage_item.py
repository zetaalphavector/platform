from datetime import datetime
from typing import Optional

from zav.pydantic_compat import BaseModel


class ObjectStorageItem(BaseModel):
    url: str
    payload: bytes


class ObjectStorageAttributes(BaseModel):
    url: str
    last_modified: datetime
    content_length: int
    content_encoding: Optional[str] = None
    content_language: Optional[str] = None
