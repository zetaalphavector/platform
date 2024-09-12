from pydantic import BaseModel


class FernetConfiguration(BaseModel):
    key: str
