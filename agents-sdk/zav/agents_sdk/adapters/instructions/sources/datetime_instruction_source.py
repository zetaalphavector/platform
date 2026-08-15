from datetime import datetime
from zoneinfo import ZoneInfo

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.instructions.instruction_source import InstructionSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class DateTimeInstructionSourceConfiguration(BaseModel):
    enabled: bool = Field(
        True,
        description="Inject the current date and time into the system prompt.",
    )
    timezone: str = Field(
        "UTC",
        description="IANA timezone name (e.g. 'Europe/Amsterdam', 'US/Eastern').",
    )


class DateTimeInstructionSource(InstructionSource):

    source_name = "datetime"

    def __init__(self, enabled: bool, timezone: str):
        self.enabled = enabled
        self.__timezone = timezone

    async def to_prompt(self) -> str:
        tz = ZoneInfo(self.__timezone)
        now = datetime.now(tz)
        date_time_str = now.strftime("%A, %B %-d, %Y, %H:%M %Z")
        return (
            "## Current date and time\n\n"
            f"The current date and time is {date_time_str}."
        )


class DateTimeInstructionSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        datetime_instruction_source_configuration: (
            DateTimeInstructionSourceConfiguration
        ) = DateTimeInstructionSourceConfiguration(),
    ) -> DateTimeInstructionSource:
        return DateTimeInstructionSource(
            enabled=datetime_instruction_source_configuration.enabled,
            timezone=datetime_instruction_source_configuration.timezone,
        )
