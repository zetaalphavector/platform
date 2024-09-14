from typing import Optional, cast

from langchain_anthropic import ChatAnthropic
from zav.llm_domain import (
    AnthropicConfiguration,
    LLMClientConfiguration,
    LLMProviderName,
)

from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class ChatAnthropicFactory(AgentDependencyFactory):
    @classmethod
    def create(cls, config: LLMClientConfiguration) -> ChatAnthropic:
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
