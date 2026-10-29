from typing import Any, Dict, List, Optional, Set

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.interjections.interjection_source import (
    InterjectionSource,
    InterjectionSourceGroup,
)
from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
)
from zav.agents_sdk.adapters.llm_models.chat_turn_request import (
    InnerLoopContext,
    InnerLoopHooks,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_INTERJECTIONS_PROMPT = """\
## Interjections

While you work through a task, the harness may add developer messages marked \
`[interjection: <source>]` between your tool calls. They come from supervisors \
watching your progress, not from the user. Treat them as advisory: weigh the \
observation, change course if it is warranted, and keep the user's request as \
the objective.\
"""


class InterjectionsProviderConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable interjections between tool turns.")
    include_sources: Optional[List[str]] = Field(
        None, description="Allowlist of interjection source names to include."
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of interjection source names to exclude."
    )


class InterjectionsProvider(InnerLoopHooks):
    """The agent's inner-loop hooks: before every model call and after every
    model response it asks each active source whether it has something to say
    and appends the answers as developer messages. With no active source the
    agent gets ``hooks is None`` and the runner skips both seams entirely."""

    def __init__(
        self,
        sources: List[InterjectionSource],
        enabled: bool,
        include_sources: Optional[Set[str]],
        exclude_sources: Optional[Set[str]],
    ) -> None:
        self.__sources = sources
        self.__enabled = enabled
        self.__include_sources = include_sources
        self.__exclude_sources = exclude_sources
        self.__cached_active: Optional[List[InterjectionSource]] = None

    def describe_loaded(self) -> Dict[str, Any]:
        return {
            "enabled": self.__enabled,
            "sources": [s.source_name for s in self.__active_sources()],
        }

    async def to_prompt(self) -> str:
        if not self.__active_sources():
            return ""
        return _INTERJECTIONS_PROMPT

    @property
    def hooks(self) -> Optional[InnerLoopHooks]:
        return self if self.__active_sources() else None

    def source(self, name: str) -> Optional[InterjectionSource]:
        return next((s for s in self.__active_sources() if s.source_name == name), None)

    async def before_model_call(
        self, context: InnerLoopContext
    ) -> List[ChatCompletion]:
        return await self.__collect("before_model_call", context)

    async def after_model_call(self, context: InnerLoopContext) -> List[ChatCompletion]:
        return await self.__collect("after_model_call", context)

    async def __collect(
        self, seam: str, context: InnerLoopContext
    ) -> List[ChatCompletion]:
        interjections: List[ChatCompletion] = []
        for source in self.__active_sources():
            try:
                text = await getattr(source, seam)(context)
            except Exception as e:
                logger.error(f"Interjection source '{source.source_name}' failed: {e}")
                continue
            if not text:
                continue
            interjections.append(
                ChatCompletion(
                    sender=ChatCompletionSender.DEVELOPER,
                    content=f"[interjection: {source.source_name}]\n{text}",
                )
            )
        return interjections

    def __active_sources(self) -> List[InterjectionSource]:
        if self.__cached_active is not None:
            return self.__cached_active
        self.__cached_active = (
            [
                source
                for source in self.__sources
                if is_source_active(
                    source.source_name,
                    source.enabled,
                    self.__include_sources,
                    self.__exclude_sources,
                )
            ]
            if self.__enabled
            else []
        )
        return self.__cached_active


class InterjectionsProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        interjection_source_group: InterjectionSourceGroup = InterjectionSourceGroup(
            items=[]
        ),
        interjections_provider_configuration: InterjectionsProviderConfiguration = (
            InterjectionsProviderConfiguration()
        ),
    ) -> InterjectionsProvider:
        return InterjectionsProvider(
            sources=interjection_source_group.items,
            enabled=interjections_provider_configuration.enabled,
            include_sources=(
                set(interjections_provider_configuration.include_sources)
                if interjections_provider_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(interjections_provider_configuration.exclude_sources)
                if interjections_provider_configuration.exclude_sources
                else None
            ),
        )
