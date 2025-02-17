from datetime import datetime
from typing import Optional

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore


class ObjectStorageItem(BaseModel):
    url: str
    payload: bytes


class ObjectStorageAttributes(BaseModel):
    url: str
    last_modified: datetime
    content_length: int
    content_encoding: Optional[str] = None
    content_language: Optional[str] = None
