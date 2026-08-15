from enum import Enum
from typing import Literal, Optional, Union, get_args

from typing_extensions import TypedDict
from zav.encryption.pydantic import EncryptedStr
from zav.pydantic_compat import (
    PYDANTIC_V2,
    BaseModel,
    ConfigDict,
    Field,
    root_validator,
)


class LLMModelType(str, Enum):
    CHAT = "chat"
    PROMPT = "prompt"
    PROMPT_WITH_LOGITS = "prompt_with_logits"
    RESPONSES = "responses"


class LLMProviderName(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    AZURE_ANTHROPIC = "azure_anthropic"
    BEDROCK = "bedrock"


class ToolChoiceNoneHandling(str, Enum):
    OMIT_TOOLS = "omit_tools"


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


class BedrockConfiguration(BaseModel):
    aws_region: str
    aws_access_key: Optional[EncryptedStr] = None
    aws_secret_key: Optional[EncryptedStr] = None
    endpoint_url: Optional[str] = None


class AzureOpenAIApiKeyAuth(BaseModel):
    api_key: EncryptedStr


class AzureOpenAIClientSecretAuth(BaseModel):
    tenant_id: str
    client_id: str
    client_secret: EncryptedStr


class AzureOpenAIWorkloadIdentityAuth(BaseModel):
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None


class AzureOpenAIConfiguration(BaseModel):
    endpoint: str
    api_version: str
    auth_type: Literal["api_key", "client_secret", "workload_identity"]
    api_key: Optional[AzureOpenAIApiKeyAuth] = None
    client_secret: Optional[AzureOpenAIClientSecretAuth] = None
    workload_identity: Optional[AzureOpenAIWorkloadIdentityAuth] = None

    @root_validator()
    @classmethod
    def auth_matches_type(cls, values):
        if PYDANTIC_V2:
            vals = values.model_dump()
        else:
            vals = values
        auth_type = vals.get("auth_type")
        if not auth_type:
            return values
        if not vals.get(auth_type):
            raise ValueError(
                f"auth_type is '{auth_type}' but '{auth_type}' is not set."
            )
        for other in get_args(cls.__annotations__["auth_type"]):
            if other != auth_type and vals.get(other):
                raise ValueError(
                    f"auth_type is '{auth_type}' but '{other}' is also set."
                )
        return values


class AzureAnthropicConfiguration(BaseModel):
    endpoint: str
    auth_type: Literal["api_key", "client_secret", "workload_identity"]
    api_key: Optional[AzureOpenAIApiKeyAuth] = None
    client_secret: Optional[AzureOpenAIClientSecretAuth] = None
    workload_identity: Optional[AzureOpenAIWorkloadIdentityAuth] = None

    @root_validator()
    @classmethod
    def auth_matches_type(cls, values):
        if PYDANTIC_V2:
            vals = values.model_dump()
        else:
            vals = values
        auth_type = vals.get("auth_type")
        if not auth_type:
            return values
        if not vals.get(auth_type):
            raise ValueError(
                f"auth_type is '{auth_type}' but '{auth_type}' is not set."
            )
        for other in get_args(cls.__annotations__["auth_type"]):
            if other != auth_type and vals.get(other):
                raise ValueError(
                    f"auth_type is '{auth_type}' but '{other}' is also set."
                )
        return values


class LLMVendorConfiguration(BaseModel):
    openai: Optional[OpenAIConfiguration] = None
    anthropic: Optional[AnthropicConfiguration] = None
    azure_openai: Optional[AzureOpenAIConfiguration] = None
    azure_anthropic: Optional[AzureAnthropicConfiguration] = None
    bedrock: Optional[BedrockConfiguration] = None

    @root_validator()
    @classmethod
    def one_of(cls, values):
        """Verify it's just one of the fields."""
        if PYDANTIC_V2:
            vals = values.model_dump()
        else:
            vals = values
        if len([val for val in vals.values() if val]) > 1:
            raise ValueError("Only one field must have a value.")
        return values


class LLMModelConfiguration(BaseModel):
    name: str
    type: LLMModelType
    temperature: Optional[float] = None
    json_output: bool = False
    max_tokens: Optional[int] = None
    interleave_system_message: Optional[str] = None
    reasoning_effort: Optional[str] = Field(
        None, description="Constrains effort on reasoning for reasoning models."
    )
    logprobs: Optional[bool] = Field(
        None,
        description="Whether to return log probabilities of the output tokens or not.",
    )
    parallel_tool_calls: Optional[bool] = Field(
        None, description="Whether to enable parallel function calling during tool use."
    )
    seed: Optional[int] = Field(
        None,
        description="If specified, the system will make a best effort to sample "
        "deterministically, such that repeated requests with the same seed and "
        "parameters should return the same result. Determinism is not guaranteed.",
    )
    verbosity: Optional[str] = Field(
        None, description="Constrains the verbosity of the model's response."
    )
    prompt_cache: bool = Field(
        True,
        description="Controls whether or not to use Anthropic's prompt caching. "
        "This is ignored for non-Anthropic models.",
    )
    tool_choice_none_handling: Optional[ToolChoiceNoneHandling] = Field(
        None,
        description='With "omit_tools", the tools and tool_choice are omitted when '
        "tool_choice is none, for OpenAI-compatible servers that would otherwise "
        "leak tool-call markup. Only for openai and azure_openai chat models.",
    )

    @root_validator()
    @classmethod
    def options_match_model_type(cls, values):
        if PYDANTIC_V2:
            model_type = values.type
            temperature = values.temperature
        else:
            model_type = values.get("type")
            temperature = values.get("temperature")

        if (
            model_type is not None
            and model_type != LLMModelType.RESPONSES
            and temperature is None
        ):
            raise ValueError("temperature is required for non-responses models.")
        return values


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

    @root_validator()
    @classmethod
    def provider_supports_model_type(cls, values):
        if PYDANTIC_V2:
            vendor = values.vendor
            vendor_configuration = values.vendor_configuration
            model_configuration = values.model_configuration
        else:
            vendor = values.get("vendor")
            vendor_configuration = values.get("vendor_configuration")
            model_configuration = values.get("model_configuration")

        if (
            not model_configuration
            or model_configuration.type != LLMModelType.RESPONSES
        ):
            return values

        supported_vendors = {
            LLMProviderName.OPENAI,
            LLMProviderName.AZURE_OPENAI,
        }
        if vendor not in supported_vendors:
            raise ValueError(
                "responses models are supported only for openai and azure_openai."
            )

        openai_configuration = (
            vendor_configuration.openai if vendor_configuration else None
        )
        if (
            vendor == LLMProviderName.OPENAI
            and openai_configuration
            and openai_configuration.openai_api_type == "azure"
        ):
            raise ValueError(
                "responses models using Azure must set vendor='azure_openai' "
                "instead of openai_api_type='azure'."
            )
        return values

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

    if PYDANTIC_V2:
        model_config = ConfigDict(from_attributes=True)
    else:

        class Config:
            orm_mode = True
