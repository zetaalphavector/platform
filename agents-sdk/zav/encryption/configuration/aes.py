from zav.pydantic_compat import BaseModel


class AesConfiguration(BaseModel):
    key: str
    iv_bytes: int = 16
