from zav.pydantic_compat import BaseModel


class FernetConfiguration(BaseModel):
    key: str
