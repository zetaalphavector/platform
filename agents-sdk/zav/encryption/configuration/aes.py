from pydantic import BaseModel


class AesConfiguration(BaseModel):
    key: str
    iv_bytes: int = 16
