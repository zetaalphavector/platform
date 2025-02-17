from typing import Optional

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore


class KmsConfiguration(BaseModel):
    key_id: str
    region_name: Optional[str] = None
