from enum import Enum
from typing import List, Literal, Optional, Union, get_args

from typing_extensions import TypedDict
from zav.encryption.pydantic import EncryptedStr, EncryptedStrRef, secret_or_ref
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
    REALTIME_TRANSCRIPTION = "realtime_transcription"
    IMAGE = "image"


class LLMProviderName(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    AZURE_ANTHROPIC = "azure_anthropic"
    BEDROCK = "bedrock"


class ToolChoiceNoneHandling(str, Enum):
    OMIT_TOOLS = "omit_tools"


class ToolSearchMode(str, Enum):
    # The vendor searches the deferred tools (OpenAI Responses `tool_search`,
    # Anthropic's tool search tool).
    SERVER = "server"
    # The vendor's tool search protocol, our search answers it.
    CLIENT = "client"
    # Any chat model: a `search_tools` function loads deferred tools.
    EMULATED = "emulated"


class ToolSearchCatalogue(str, Enum):
    GROUPS = "groups"
    NAMES = "names"
    DESCRIPTIONS = "descriptions"


class ToolSearchOptions(BaseModel):
    catalogue: ToolSearchCatalogue = Field(
        ToolSearchCatalogue.NAMES,
        description="How much the search tool's description lists of the deferred "
        "tools: their groups, the groups with tool names, or every tool with a "
        "one-line description. Ignored by `server`, where the vendor lists them.",
    )
    max_results: int = Field(
        8,
        ge=1,
        description="Most tools one search loads by query. Ignored by `server`.",
    )


class ToolSearchMethodSettings(ToolSearchOptions):
    method: ToolSearchMode


class AnthropicConfiguration(BaseModel):
    anthropic_api_key: Optional[EncryptedStr] = None
    anthropic_api_key_ref: Optional[EncryptedStrRef] = None
    anthropic_api_type: Optional[str] = None
    anthropic_api_base: Optional[str] = None
    aws_secret_key: Optional[EncryptedStr] = None
    aws_secret_key_ref: Optional[EncryptedStrRef] = None
    aws_access_key: Optional[EncryptedStr] = None
    aws_access_key_ref: Optional[EncryptedStrRef] = None
    aws_region: Optional[str] = None

    @root_validator()
    @classmethod
    def secrets_or_refs(cls, values):
        vals = values.model_dump() if PYDANTIC_V2 else values
        secret_or_ref(vals, "anthropic_api_key", required=True)
        secret_or_ref(vals, "aws_secret_key")
        secret_or_ref(vals, "aws_access_key")
        return values


class OpenAIConfiguration(BaseModel):
    openai_api_key: Optional[EncryptedStr] = None
    openai_api_key_ref: Optional[EncryptedStrRef] = None
    openai_org: Optional[EncryptedStr] = None
    openai_org_ref: Optional[EncryptedStrRef] = None
    openai_api_type: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None

    @root_validator()
    @classmethod
    def secrets_or_refs(cls, values):
        vals = values.model_dump() if PYDANTIC_V2 else values
        secret_or_ref(vals, "openai_api_key", required=True)
        secret_or_ref(vals, "openai_org")
        return values


class BedrockConfiguration(BaseModel):
    aws_region: str
    aws_access_key: Optional[EncryptedStr] = None
    aws_access_key_ref: Optional[EncryptedStrRef] = None
    aws_secret_key: Optional[EncryptedStr] = None
    aws_secret_key_ref: Optional[EncryptedStrRef] = None
    endpoint_url: Optional[str] = None

    @root_validator()
    @classmethod
    def secrets_or_refs(cls, values):
        vals = values.model_dump() if PYDANTIC_V2 else values
        secret_or_ref(vals, "aws_access_key")
        secret_or_ref(vals, "aws_secret_key")
        return values


class AzureOpenAIApiKeyAuth(BaseModel):
    api_key: Optional[EncryptedStr] = None
    api_key_ref: Optional[EncryptedStrRef] = None

    @root_validator()
    @classmethod
    def secrets_or_refs(cls, values):
        vals = values.model_dump() if PYDANTIC_V2 else values
        secret_or_ref(vals, "api_key", required=True)
        return values


class AzureOpenAIClientSecretAuth(BaseModel):
    tenant_id: str
    client_id: str
    client_secret: Optional[EncryptedStr] = None
    client_secret_ref: Optional[EncryptedStrRef] = None

    @root_validator()
    @classmethod
    def secrets_or_refs(cls, values):
        vals = values.model_dump() if PYDANTIC_V2 else values
        secret_or_ref(vals, "client_secret", required=True)
        return values


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


def takes_single_language(model_name: str) -> bool:
    # whisper-1 and the gpt-4o transcribe models take one `language`;
    # gpt-live-transcribe and later take `languages`. On Azure the name is the
    # deployment name, which by convention contains the model name.
    name = model_name.lower()
    return "whisper" in name or ("4o" in name and "transcribe" in name)


class TranscriptionSettings(BaseModel):
    languages: Optional[List[str]] = Field(
        None,
        description="Language hints such as ['nl', 'en']: ISO 639-1, selected "
        "ISO 639-3 or zh locale codes. whisper-1 and the gpt-4o transcribe models "
        "take one. Unset lets the model detect them.",
    )
    max_session_seconds: int = Field(
        300,
        gt=0,
        description="Wall-clock cap per session.",
    )


class ImageQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"
    AUTO = "auto"


class ImageOutputFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class ImageModeration(str, Enum):
    AUTO = "auto"
    LOW = "low"


class ImageSettings(BaseModel):
    """Unset fields leave the vendor default in place."""

    quality: Optional[ImageQuality] = Field(
        None, description="Rendering quality, in OpenAI's levels."
    )
    output_format: Optional[ImageOutputFormat] = Field(
        None,
        description="Encoding of generated images. jpeg switches to png when a "
        "transparent background is requested.",
    )
    output_compression: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Compression level (0-100), sent with jpeg and webp output.",
    )
    moderation: Optional[ImageModeration] = Field(
        None, description="Content moderation strictness."
    )


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
    tool_search_methods: Optional[List[ToolSearchMode]] = Field(
        None,
        description="The tool search methods this model supports; the agent uses "
        "the first of its enabled methods that is listed. `server` and `client` "
        "need the vendor's tool search (OpenAI responses models, Anthropic chat "
        "models). Unset: `emulated` only, for chat and responses models.",
    )
    transcription_settings: Optional[TranscriptionSettings] = Field(
        None,
        description="Options of realtime transcription models. Only for "
        "realtime_transcription models.",
    )
    image_settings: Optional[ImageSettings] = Field(
        None,
        description="Rendering options of image models. Only for image models.",
    )

    @root_validator()
    @classmethod
    def options_match_model_type(cls, values):
        if PYDANTIC_V2:
            vals = values.model_dump()
        else:
            vals = values
        model_type = vals.get("type")
        if (
            model_type is not None
            and model_type
            not in (
                LLMModelType.RESPONSES,
                LLMModelType.REALTIME_TRANSCRIPTION,
                LLMModelType.IMAGE,
            )
            and vals.get("temperature") is None
        ):
            raise ValueError(
                "temperature is required unless the model type is responses, "
                "realtime_transcription or image."
            )
        if (
            model_type == LLMModelType.REALTIME_TRANSCRIPTION
            and vals.get("temperature") is not None
        ):
            raise ValueError(
                "temperature cannot be set for realtime_transcription models."
            )
        transcription_settings = vals.get("transcription_settings")
        if (
            transcription_settings is not None
            and model_type != LLMModelType.REALTIME_TRANSCRIPTION
        ):
            raise ValueError(
                "transcription_settings can only be set for realtime_transcription "
                "models."
            )
        languages = (
            transcription_settings.get("languages")
            if isinstance(transcription_settings, dict)
            else getattr(transcription_settings, "languages", None)
        )
        if (
            languages
            and len(languages) > 1
            and takes_single_language(vals.get("name") or "")
        ):
            raise ValueError(f"{vals.get('name')} takes one language.")
        if vals.get("image_settings") is not None and model_type != LLMModelType.IMAGE:
            raise ValueError("image_settings can only be set for image models.")
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


def _searches_natively(vendor: LLMProviderName, model_type: LLMModelType) -> bool:
    return (
        model_type == LLMModelType.RESPONSES
        and vendor in (LLMProviderName.OPENAI, LLMProviderName.AZURE_OPENAI)
    ) or (
        model_type == LLMModelType.CHAT
        and vendor in (LLMProviderName.ANTHROPIC, LLMProviderName.AZURE_ANTHROPIC)
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
            model_configuration
            and {ToolSearchMode.SERVER, ToolSearchMode.CLIENT}
            & set(model_configuration.tool_search_methods or [])
            and not _searches_natively(vendor, model_configuration.type)
        ):
            raise ValueError(
                "tool_search_methods server and client need an openai or "
                "azure_openai responses model or an anthropic or azure_anthropic "
                "chat model; use emulated for other models."
            )

        if not model_configuration or model_configuration.type not in (
            LLMModelType.RESPONSES,
            LLMModelType.REALTIME_TRANSCRIPTION,
            LLMModelType.IMAGE,
        ):
            return values

        supported_vendors = {
            LLMProviderName.OPENAI,
            LLMProviderName.AZURE_OPENAI,
        }
        if vendor not in supported_vendors:
            raise ValueError(
                f"{model_configuration.type.value} models are supported only for "
                "openai and azure_openai."
            )
        if model_configuration.type == LLMModelType.IMAGE:
            return values

        openai_configuration = (
            vendor_configuration.openai if vendor_configuration else None
        )
        if (
            vendor == LLMProviderName.OPENAI
            and openai_configuration
            and openai_configuration.openai_api_type == "azure"
        ):
            raise ValueError(
                f"{model_configuration.type.value} models using Azure must set "
                "vendor='azure_openai' instead of openai_api_type='azure'."
            )
        return values

    def supported_tool_search_methods(self) -> List[ToolSearchMode]:
        if self.model_configuration.type not in (
            LLMModelType.CHAT,
            LLMModelType.RESPONSES,
        ):
            return []
        if self.model_configuration.tool_search_methods is None:
            return [ToolSearchMode.EMULATED]
        return list(self.model_configuration.tool_search_methods)

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
