from enum import Enum
from typing import Literal, Optional

from zav.encryption.pydantic import EncryptedStr, EncryptedStrRef, secret_or_ref
from zav.pydantic_compat import (
    PYDANTIC_V2,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    root_validator,
)

from zav.llm_tracing.local_trace_store import LocalTraceStore


class LangfuseConfiguration(BaseModel):
    """Fields:
    public_key: Public API key of Langfuse project.
    secret_key: Secret API key of Langfuse project.
    secret_key_ref: Name of the tenant secret that holds the secret key instead.
    host: Host of Langfuse API. Defaults to `https://cloud.langfuse.com`.
    release: Release number/hash of the application to provide analytics
        grouped by release.
    debug: Enables debug mode for more verbose logging.
    threads: Number of consumer threads to execute network requests.
        Helps scaling the SDK for high load. Only increase this if you run
        into scaling issues.
    flush_at: Max batch size that's sent to the API.
    flush_interval: Max delay until a new batch is sent to the API.
    max_retries: Max number of retries in case of API/network errors.
    timeout: Timeout of API requests in seconds. Defaults to 20 seconds.
    httpx_client: Pass your own httpx client for more customizability
        of requests.
    sdk_integration: Used by intgerations that wrap the Langfuse SDK to
        add context for debugging and support. Not to be used directly.
    enabled: Enables or disables the Langfuse client.
    sample_rate: Sampling rate for tracing. If set to 0.2, only 20% of the
        data will be sent to the backend."""

    host: str
    secret_key: Optional[EncryptedStr] = None
    secret_key_ref: Optional[EncryptedStrRef] = None
    public_key: str
    enabled: bool = True
    release: Optional[str] = None
    debug: bool = False
    threads: Optional[int] = None
    flush_at: Optional[int] = None
    flush_interval: Optional[float] = None
    max_retries: Optional[int] = None
    timeout: Optional[int] = None  # seconds
    sdk_integration: Optional[str] = "default"
    sample_rate: Optional[float] = None

    @root_validator()
    @classmethod
    def secrets_or_refs(cls, values):
        vals = values.model_dump() if PYDANTIC_V2 else values
        secret_or_ref(vals, "secret_key", required=True)
        return values


class CaptureConfiguration(BaseModel):
    _store: LocalTraceStore = PrivateAttr(default_factory=LocalTraceStore)

    @property
    def store(self) -> LocalTraceStore:
        return self._store

    def __init__(self, **data):
        # Extract _store or store if provided, otherwise use default
        _store = data.pop("_store", None)
        store = data.pop("store", None)
        super().__init__(**data)
        # Use _store first, then store, then default
        self._store = _store or store or LocalTraceStore()


class FileConfiguration(BaseModel):
    """Fields:
    directory: Directory the traces are written to, one file per trace in the
        `LocalTraceStore` export format. Relative paths resolve against the
        working directory.
    file_name: `timestamp_and_trace_id` writes `<start time>_<trace id>.trace.json`,
        so every run of a trace id leaves its own file; `trace_id` writes
        `<trace id>.trace.json`, so a rerun replaces the earlier file."""

    directory: str
    file_name: Literal["timestamp_and_trace_id", "trace_id"] = "timestamp_and_trace_id"


class TracingVendorConfiguration(BaseModel):
    langfuse: Optional[LangfuseConfiguration] = None
    capture: Optional[CaptureConfiguration] = None
    file: Optional[FileConfiguration] = None


class TracingVendorName(str, Enum):
    LANGFUSE = "langfuse"
    CAPTURE = "capture"
    FILE = "file"


class TracingConfiguration(BaseModel):
    vendor: TracingVendorName
    vendor_configuration: TracingVendorConfiguration = Field(
        default_factory=TracingVendorConfiguration
    )

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True
