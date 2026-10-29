from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import (
    ClientSecretCredential,
    WorkloadIdentityCredential,
    get_bearer_token_provider,
)
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from zav.encryption.pydantic import EncryptedStr
from zav.llm_domain import (
    AzureOpenAIConfiguration,
    LLMClientConfiguration,
    LLMModelType,
    LLMProviderName,
    OpenAIConfiguration,
    TranscriptionSettings,
    takes_single_language,
)

from zav.agents_sdk.adapters.realtime_transcription.session import (
    RealtimeTranscriptionSession,
    session_update,
)

_AZURE_SCOPE = "https://cognitiveservices.azure.com/.default"


def _secret(value: Optional[EncryptedStr], field: str) -> str:
    secret = value.get_unencrypted_secret() if value is not None else None
    if secret is None:
        raise ValueError(f"{field} is not configured")
    return secret


def _openai_client(configuration: OpenAIConfiguration) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_secret(configuration.openai_api_key, "openai_api_key"),
        organization=(
            configuration.openai_org.get_unencrypted_secret()
            if configuration.openai_org is not None
            else None
        ),
        base_url=configuration.openai_api_base,
    )


def _azure_credential(configuration: AzureOpenAIConfiguration) -> AsyncTokenCredential:
    if configuration.auth_type == "client_secret":
        assert configuration.client_secret is not None
        return ClientSecretCredential(
            tenant_id=configuration.client_secret.tenant_id,
            client_id=configuration.client_secret.client_id,
            client_secret=_secret(
                configuration.client_secret.client_secret, "client_secret"
            ),
        )
    identity = configuration.workload_identity
    kwargs = {}
    if identity is not None and identity.tenant_id:
        kwargs["tenant_id"] = identity.tenant_id
    if identity is not None and identity.client_id:
        kwargs["client_id"] = identity.client_id
    return WorkloadIdentityCredential(**kwargs)


def _azure_openai_client(
    configuration: AzureOpenAIConfiguration,
) -> Tuple[AsyncOpenAI, Optional[AsyncTokenCredential]]:
    # Realtime transcription is served by the v1 API, not the dated
    # api-version paths AsyncAzureOpenAI builds.
    endpoint = configuration.endpoint.rstrip("/")
    if not endpoint.endswith("/openai/v1"):
        endpoint = f"{endpoint}/openai/v1"
    if configuration.auth_type == "api_key":
        assert configuration.api_key is not None
        api_key = _secret(configuration.api_key.api_key, "api_key")
        return AsyncOpenAI(base_url=endpoint, api_key=api_key), None
    # AsyncOpenAI awaits a callable api_key, so the token provider must be async.
    credential = _azure_credential(configuration)
    return (
        AsyncOpenAI(
            base_url=endpoint,
            api_key=get_bearer_token_provider(credential, _AZURE_SCOPE),
        ),
        credential,
    )


class OpenAIRealtimeTranscription:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_session_seconds: float,
        languages: Optional[List[str]] = None,
        credential: Optional[AsyncTokenCredential] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.__client = client
        self.__model = model
        self.__max_session_seconds = max_session_seconds
        self.__language_hint: Dict[str, Any] = (
            {}
            if not languages
            else (
                {"language": languages[0]}
                if takes_single_language(model)
                else {"languages": languages}
            )
        )
        self.__credential = credential
        self.__clock = clock

    @classmethod
    def from_llm_configuration(
        cls, configuration: LLMClientConfiguration
    ) -> "OpenAIRealtimeTranscription":
        model_configuration = configuration.model_configuration
        if model_configuration.type != LLMModelType.REALTIME_TRANSCRIPTION:
            raise ValueError(
                f"LLM configuration type '{model_configuration.type.value}' is not "
                "realtime_transcription."
            )
        settings = model_configuration.transcription_settings or TranscriptionSettings()
        vendor_configuration = configuration.vendor_configuration
        credential: Optional[AsyncTokenCredential] = None
        if (
            configuration.vendor == LLMProviderName.OPENAI
            and vendor_configuration.openai is not None
        ):
            client = _openai_client(vendor_configuration.openai)
        elif (
            configuration.vendor == LLMProviderName.AZURE_OPENAI
            and vendor_configuration.azure_openai is not None
        ):
            client, credential = _azure_openai_client(vendor_configuration.azure_openai)
        else:
            raise ValueError(
                f"vendor_configuration.{configuration.vendor.value} is not set."
            )
        return cls(
            client=client,
            model=model_configuration.name,
            max_session_seconds=settings.max_session_seconds,
            languages=settings.languages,
            credential=credential,
        )

    @asynccontextmanager
    async def open_session(self) -> AsyncIterator[RealtimeTranscriptionSession]:
        connection: Optional[AsyncRealtimeConnection] = None
        try:
            # A transcription session is opened by intent; the model goes in
            # session.update. OpenAI rejects ?model=<transcription model>.
            connection = await self.__client.realtime.connect(
                extra_query={"intent": "transcription"}
            ).enter()
            await connection.send(
                session_update(  # type: ignore[arg-type]
                    self.__model, self.__language_hint
                )
            )
            yield RealtimeTranscriptionSession(
                connection,
                self.__max_session_seconds,
                self.__model,
                self.__language_hint,
                self.__clock,
            )
        finally:
            try:
                if connection is not None:
                    await connection.close()
            finally:
                await self.__client.close()
                if self.__credential is not None:
                    await self.__credential.close()
