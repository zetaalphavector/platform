from typing import Optional

from zav.pydantic_compat import BaseModel, Field


class LLMSelectionConfiguration(BaseModel):
    """The model policy: selects a named LLM configuration for the request.
    Carries a name only; it is honored when the setup allows it."""

    llm_configuration_name: Optional[str] = Field(
        None,
        description="Named LLM configuration to run this request on.",
    )
