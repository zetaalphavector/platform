from typing import Optional, cast

from langchain_openai import ChatOpenAI
from zav.llm_domain import LLMClientConfiguration, LLMProviderName, OpenAIConfiguration

from zav.agents_sdk.adapters.policies.llm_selection import LLMSelectionConfiguration
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.llm_client_factory import LLMClientFactory, LLMNotConfigured


def build_chat_openai(config: LLMClientConfiguration) -> ChatOpenAI:
    if (
        config.vendor == LLMProviderName.OPENAI
        or config.vendor == LLMProviderName.OLLAMA
    ) and (
        openai_config := cast(
            Optional[OpenAIConfiguration],
            getattr(config.vendor_configuration, config.vendor.value, None),
        )
    ):
        return ChatOpenAI(
            openai_api_key=(  # type: ignore
                openai_config.openai_api_key.get_unencrypted_secret()
            ),
            openai_organization=(  # type: ignore
                openai_config.openai_org.get_unencrypted_secret()
            ),
            openai_api_base=openai_config.openai_api_base,  # type: ignore
            model=config.model_configuration.name,
            temperature=config.model_configuration.temperature,
            max_tokens=config.model_configuration.max_tokens,
        )

    raise ValueError(f"Unsupported vendor: {config.vendor}")


class ChatOpenAIFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        llm_client_factory: Optional[LLMClientFactory] = None,
        llm_selection_configuration: LLMSelectionConfiguration = (
            LLMSelectionConfiguration()
        ),
    ) -> ChatOpenAI:
        if llm_client_factory is None:
            raise LLMNotConfigured("No LLM client factory in scope.")
        return llm_client_factory.create_client(
            build_chat_openai,
            requested=llm_selection_configuration.llm_configuration_name,
        )
