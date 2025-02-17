try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore


class AesConfiguration(BaseModel):
    key: str
    iv_bytes: int = 16
