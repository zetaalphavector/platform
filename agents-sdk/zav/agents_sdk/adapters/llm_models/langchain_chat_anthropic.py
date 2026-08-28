from typing import Optional, cast

from langchain_anthropic import ChatAnthropic
from zav.llm_domain import (
    AnthropicConfiguration,
    LLMClientConfiguration,
    LLMProviderName,
)

from zav.agents_sdk.adapters.policies.llm_selection import LLMSelectionConfiguration
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.llm_client_factory import LLMClientFactory, LLMNotConfigured


def build_chat_anthropic(config: LLMClientConfiguration) -> ChatAnthropic:
    if config.vendor == LLMProviderName.ANTHROPIC and (
        anthropic_config := cast(
            Optional[AnthropicConfiguration],
            getattr(config.vendor_configuration, config.vendor.value, None),
        )
    ):
        if anthropic_config.anthropic_api_type != "bedrock":
            return ChatAnthropic(
                anthropic_api_key=(  # type: ignore
                    anthropic_config.anthropic_api_key.get_unencrypted_secret()
                ),
                anthropic_api_url=anthropic_config.anthropic_api_base,
                model=config.model_configuration.name,
                temperature=config.model_configuration.temperature,
                max_tokens=config.model_configuration.max_tokens,
            )

    raise ValueError(f"Unsupported vendor: {config.vendor}")


class ChatAnthropicFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        llm_client_factory: Optional[LLMClientFactory] = None,
        llm_selection_configuration: LLMSelectionConfiguration = (
            LLMSelectionConfiguration()
        ),
    ) -> ChatAnthropic:
        if llm_client_factory is None:
            raise LLMNotConfigured("No LLM client factory in scope.")
        return llm_client_factory.create_client(
            build_chat_anthropic,
            requested=llm_selection_configuration.llm_configuration_name,
        )
