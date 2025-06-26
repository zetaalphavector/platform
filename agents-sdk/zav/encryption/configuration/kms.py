from typing import Optional

from zav.pydantic_compat import BaseModel


class KmsConfiguration(BaseModel):
    key_id: str
    region_name: Optional[str] = None
