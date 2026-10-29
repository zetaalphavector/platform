import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from uuid import uuid4

from zav.pydantic_compat import BaseModel


class RealtimeTranscriptionNotConfigured(Exception):
    pass


def error_event(error_type: str, code: str, message: str) -> str:
    # An OpenAI Realtime `error` event, so clients handle ours like the provider's.
    return json.dumps(
        {
            "type": "error",
            "event_id": f"event_{uuid4().hex}",
            "error": {"type": error_type, "code": code, "message": message},
        }
    )


class RealtimeTranscriptionSetup(BaseModel):
    # A realtime_transcription entry of the host's LLM configurations.
    llm_configuration_name: str


class RealtimeTranscriptionSetupRetriever(ABC):
    # Tenant-parameterized like MCPServerSetupRetriever: hosted retrievers read
    # the tenant's settings; the file retriever ignores the tenant.
    @abstractmethod
    async def get(self, tenant: str) -> Optional[RealtimeTranscriptionSetup]:
        raise NotImplementedError


class FileRealtimeTranscriptionSetupRetriever(RealtimeTranscriptionSetupRetriever):
    def __init__(self, file_path: str):
        self.__file_path = file_path

    async def get(self, tenant: str) -> Optional[RealtimeTranscriptionSetup]:
        # Re-read on every session so local edits apply without a restart.
        path = Path(self.__file_path)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed realtime transcription setup file {path}: {error}"
            ) from error
        return RealtimeTranscriptionSetup(**raw)
