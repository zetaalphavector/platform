from typing import Optional, Union

import anthropic
from azure.identity import (
    ClientSecretCredential,
    WorkloadIdentityCredential,
    get_bearer_token_provider,
)
from zav.llm_domain import (
    AzureAnthropicConfiguration,
    LLMModelConfiguration,
    LLMModelType,
    LLMProviderName,
)
from zav.llm_tracing import Span

from zav.prompt_completion.adapters.anthropic_clients import (
    AnthropicChatClient,
    AnthropicChatClient2PromptClientAdapter,
)
from zav.prompt_completion.client_factories import (
    ChatClientFactory,
    PromptClientFactory,
)


def build_client(
    vendor_configuration: AzureAnthropicConfiguration,
) -> anthropic.AsyncAnthropicFoundry:
    if vendor_configuration.auth_type == "api_key":
        assert vendor_configuration.api_key is not None
        return anthropic.AsyncAnthropicFoundry(
            api_key=vendor_configuration.api_key.api_key.get_unencrypted_secret(),
            base_url=vendor_configuration.endpoint,
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
    return anthropic.AsyncAnthropicFoundry(
        azure_ad_token_provider=token_provider,
        base_url=vendor_configuration.endpoint,
    )


@ChatClientFactory.register(LLMProviderName.AZURE_ANTHROPIC, LLMModelType.CHAT)
class AzureAnthropicChatClient(AnthropicChatClient):
    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: AzureAnthropicConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "AzureAnthropicChatClient":
        client = build_client(vendor_configuration)
        return cls(client, model_configuration, span=span)


@PromptClientFactory.register(LLMProviderName.AZURE_ANTHROPIC, LLMModelType.CHAT)
class AzureAnthropicChatClient2PromptClientAdapter(
    AnthropicChatClient2PromptClientAdapter,
):
    @classmethod
    def from_configuration(
        cls,
        vendor_configuration: AzureAnthropicConfiguration,
        model_configuration: LLMModelConfiguration,
        span: Optional[Span] = None,
    ) -> "AzureAnthropicChatClient2PromptClientAdapter":
        chat_client = AzureAnthropicChatClient.from_configuration(
            vendor_configuration, model_configuration, span=span
        )
        return cls(chat_client=chat_client)
