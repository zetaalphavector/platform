from typing import Optional

from pydantic import BaseModel


class KmsConfiguration(BaseModel):
    key_id: str
    region_name: Optional[str] = None
