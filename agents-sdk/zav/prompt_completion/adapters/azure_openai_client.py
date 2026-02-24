from typing import Optional, Union

import openai
from azure.identity import (
    ClientSecretCredential,
    WorkloadIdentityCredential,
    get_bearer_token_provider,
)
from zav.llm_domain import (
    AzureOpenAIConfiguration,
    LLMModelConfiguration,
    LLMModelType,
    LLMProviderName,
)
from zav.llm_tracing import Span

from zav.prompt_completion.adapters.openai_clients import (
    OpenAiChatClient,
    OpenAiChatClient2PromptClientAdapter,
    OpenAiPromptClient,
    OpenAiPromptWithLogitsClient,
)
from zav.prompt_completion.client_factories import (
    ChatClientFactory,
    PromptClientFactory,
    PromptWithLogitsClientFactory,
)


def build_client(
    vendor_configuration: AzureOpenAIConfiguration,
) -> openai.AsyncAzureOpenAI:
    if vendor_configuration.auth_type == "api_key":
        assert vendor_configuration.api_key is not None
        return openai.AsyncAzureOpenAI(
            api_key=vendor_configuration.api_key.api_key.get_unencrypted_secret(),
            azure_endpoint=vendor_configuration.endpoint,
            api_version=vendor_configuration.api_version,
        )

    if vendor_configuration.auth_type == "client_secret":
        if vendor_configuration.client_secret is None:
            raise ValueError(
                "client_secret auth_type requires client_secret configuration"
            )
        creds = vendor_configuration.client_secret
        credential: Union[ClientSecretCredential, WorkloadIdentityCredential] = (
            ClientSecretCredential(
                tenant_id=creds.tenant_id,
                client_id=creds.client_id,
                client_secret=creds.client_secret.get_unencrypted_secret(),
            )
        )
    elif vendor_configuration.auth_type == "workload_identity":
        kwargs: dict = {}
        if vendor_configuration.workload_identity is None:
            raise ValueError(
                "workload_identity auth_type requires workload_identity configuration"
            )
        wi = vendor_configuration.workload_identity
        if wi.tenant_id:
            kwargs["tenant_id"] = wi.tenant_id
        if wi.client_id:
            kwargs["client_id"] = wi.client_id
        credential = WorkloadIdentityCredential(**kwargs)
    else:
        raise ValueError(f"Unsupported auth_type: {vendor_configuration.auth_type}")

    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return openai.AsyncAzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=vendor_configuration.endpoint,
        api_version=vendor_configuration.api_version,
    )


@PromptWithLogitsClientFactory.register(
    LLMProviderName.AZURE_OPENAI, LLMModelType.PROMPT_WITH_LOGITS
)
class AzureOpenAiPromptWithLogitsClient(OpenAiPromptWithLogitsClient):
    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: AzureOpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "AzureOpenAiPromptWithLogitsClient":
        client = build_client(vendor_configuration)
        return cls(client=client, model_configuration=model_configuration, span=span)


@PromptClientFactory.register(
    LLMProviderName.AZURE_OPENAI, LLMModelType.PROMPT_WITH_LOGITS
)
@PromptClientFactory.register(LLMProviderName.AZURE_OPENAI, LLMModelType.PROMPT)
class AzureOpenAiPromptClient(OpenAiPromptClient):
    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: AzureOpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "AzureOpenAiPromptClient":
        client = build_client(vendor_configuration)
        return cls(client=client, model_configuration=model_configuration, span=span)


@ChatClientFactory.register(LLMProviderName.AZURE_OPENAI, LLMModelType.CHAT)
class AzureOpenAiChatClient(OpenAiChatClient):
    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: AzureOpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "AzureOpenAiChatClient":
        client = build_client(vendor_configuration)
        return cls(client=client, model_configuration=model_configuration, span=span)


@PromptClientFactory.register(LLMProviderName.AZURE_OPENAI, LLMModelType.CHAT)
class AzureOpenAiChatClient2PromptClientAdapter(
    OpenAiChatClient2PromptClientAdapter,
):
    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: AzureOpenAIConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "AzureOpenAiChatClient2PromptClientAdapter":
        chat_client = AzureOpenAiChatClient.from_configuration(
            vendor_configuration, model_configuration, span=span
        )
        return cls(chat_client=chat_client)
