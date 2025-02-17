try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore


class FernetConfiguration(BaseModel):
    key: str
