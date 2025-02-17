from enum import Enum
from typing import Literal, Optional, Union

try:
    from pydantic.v1 import BaseModel, Field, root_validator
except ImportError:
    from pydantic import BaseModel, Field, root_validator  # type: ignore

from typing_extensions import TypedDict
from zav.encryption.pydantic import EncryptedStr


class LLMModelType(str, Enum):
    CHAT = "chat"
    PROMPT = "prompt"
    PROMPT_WITH_LOGITS = "prompt_with_logits"


class LLMProviderName(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AnthropicConfiguration(BaseModel):
    anthropic_api_key: EncryptedStr
    anthropic_api_type: Optional[str] = None
    anthropic_api_base: Optional[str] = None
    aws_secret_key: Optional[EncryptedStr] = None
    aws_access_key: Optional[EncryptedStr] = None
    aws_region: Optional[str] = None


class OpenAIConfiguration(BaseModel):
    openai_api_key: EncryptedStr
    openai_org: EncryptedStr
    openai_api_type: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None


class LLMVendorConfiguration(BaseModel):
    openai: Optional[OpenAIConfiguration] = None
    anthropic: Optional[AnthropicConfiguration] = None

    @root_validator
    @classmethod
    def one_of(cls, v):
        """Verify it's just one of the fields."""
        if len([val for val in v.values() if val]) > 1:
            raise ValueError("Only one field must have a value.")
        return v


class LLMModelConfiguration(BaseModel):
    name: str
    type: LLMModelType
    temperature: float
    json_output: bool = False
    max_tokens: Optional[int] = None
    interleave_system_message: Optional[str] = None


class PromptModelParams(TypedDict):
    model_type: LLMModelType
    provider_name: LLMProviderName
    model_name: str


def _prompt_model_from(model_env_variable: str) -> PromptModelParams:
    if ":" not in model_env_variable:
        return PromptModelParams(
            model_type=LLMModelType.PROMPT_WITH_LOGITS,
            provider_name=LLMProviderName.OPENAI,
            model_name=model_env_variable,
        )
    model_aspects = ["model_type", "provider_name", "model_name"]
    model_config = {
        aspect: value
        for aspect, value in zip(model_aspects, model_env_variable.split(":"))
    }
    return PromptModelParams(
        model_type=LLMModelType(model_config["model_type"]),
        provider_name=LLMProviderName(model_config["provider_name"]),
        model_name=model_config["model_name"],
    )


class LLMClientConfiguration(BaseModel):
    vendor: LLMProviderName
    vendor_configuration: LLMVendorConfiguration = Field(
        default_factory=LLMVendorConfiguration
    )
    model_configuration: LLMModelConfiguration

    @classmethod
    def from_env_vars(
        cls,
        prompt_model_var: str,
        temperature_model_var: float,
        max_tokens_model_var: Optional[int] = None,
        interleave_system_message_model_var: Optional[
            Union[str, Literal["repeat_before_last_user_message"]]
        ] = None,
        **vendor_config_vars,
    ):
        prompt_model = _prompt_model_from(prompt_model_var)
        if prompt_model["provider_name"] == LLMProviderName.OPENAI:
            vendor_configuration = LLMVendorConfiguration(
                openai=OpenAIConfiguration(**vendor_config_vars)
            )
        else:
            raise ValueError(f"Unknown provider: {prompt_model['provider_name']}")
        return cls(
            vendor=prompt_model["provider_name"],
            vendor_configuration=vendor_configuration,
            model_configuration=LLMModelConfiguration(
                name=prompt_model["model_name"],
                type=prompt_model["model_type"],
                temperature=temperature_model_var,
                max_tokens=max_tokens_model_var,
                interleave_system_message=interleave_system_message_model_var,
            ),
        )

    class Config:
        orm_mode = True
