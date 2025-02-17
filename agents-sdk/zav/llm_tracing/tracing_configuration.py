from enum import Enum
from typing import Optional

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field  # type: ignore

from zav.encryption.pydantic import EncryptedStr


class LangfuseConfiguration(BaseModel):
    """Fields:
    public_key: Public API key of Langfuse project.
    secret_key: Secret API key of Langfuse project.
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
    secret_key: EncryptedStr
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


class TracingVendorConfiguration(BaseModel):
    langfuse: Optional[LangfuseConfiguration] = None


class TracingVendorName(str, Enum):
    LANGFUSE = "langfuse"


class TracingConfiguration(BaseModel):
    vendor: TracingVendorName
    vendor_configuration: TracingVendorConfiguration = Field(
        default_factory=TracingVendorConfiguration
    )

    class Config:
        orm_mode = True
