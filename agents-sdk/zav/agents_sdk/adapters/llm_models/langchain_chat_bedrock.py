from typing import Optional, cast

from langchain_aws import ChatBedrock
from zav.llm_domain import (
    AnthropicConfiguration,
    LLMClientConfiguration,
    LLMProviderName,
)

from zav.agents_sdk.adapters.policies.llm_selection import LLMSelectionConfiguration
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.llm_client_factory import LLMClientFactory, LLMNotConfigured


def build_chat_bedrock(config: LLMClientConfiguration) -> ChatBedrock:
    if config.vendor == LLMProviderName.ANTHROPIC and (
        anthropic_config := cast(
            Optional[AnthropicConfiguration],
            getattr(config.vendor_configuration, config.vendor.value, None),
        )
    ):
        if anthropic_config.anthropic_api_type == "bedrock":
            # TODO: ChatBedrock doesn't support aws_access_key_id and
            # aws_secret_access_key
            return ChatBedrock(
                aws_access_key_id=(  # type: ignore
                    anthropic_config.aws_access_key.get_unencrypted_secret()
                    if anthropic_config.aws_access_key
                    else None
                ),
                aws_secret_access_key=(  # type: ignore
                    anthropic_config.aws_secret_key.get_unencrypted_secret()
                    if anthropic_config.aws_secret_key
                    else None
                ),
                region_name=anthropic_config.aws_region,
                endpoint_url=anthropic_config.anthropic_api_base,
                model=config.model_configuration.name,  # type: ignore
                temperature=config.model_configuration.temperature,  # type: ignore
                max_tokens=config.model_configuration.max_tokens,  # type: ignore
            )
    raise ValueError(f"Unsupported vendor: {config.vendor}")


class ChatBedrockFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        llm_client_factory: Optional[LLMClientFactory] = None,
        llm_selection_configuration: LLMSelectionConfiguration = (
            LLMSelectionConfiguration()
        ),
    ) -> ChatBedrock:
        if llm_client_factory is None:
            raise LLMNotConfigured("No LLM client factory in scope.")
        return llm_client_factory.create_client(
            build_chat_bedrock,
            requested=llm_selection_configuration.llm_configuration_name,
        )
