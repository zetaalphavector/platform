from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters._filtering import is_source_active
from zav.agents_sdk.adapters.dispatch.dispatch_rule import (
    DispatchRule,
    DispatchRuleGroup,
    EmitFn,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage, ConversationContext


class DispatchProviderConfiguration(BaseModel):
    enabled: bool = Field(True, description="Enable dispatch rules.")
    include_sources: Optional[List[str]] = Field(
        None, description="Allowlist of dispatch rule source names to include."
    )
    exclude_sources: Optional[List[str]] = Field(
        None, description="Denylist of dispatch rule source names to exclude."
    )


class DispatchProvider:

    def __init__(
        self,
        rules: List[DispatchRule],
        enabled: bool = True,
        include_sources: Optional[Set[str]] = None,
        exclude_sources: Optional[Set[str]] = None,
    ) -> None:
        self.__rules = rules
        self.__enabled = enabled
        self.__include_sources = include_sources
        self.__exclude_sources = exclude_sources
        self.__cached_active: Optional[List[DispatchRule]] = None

    def describe_loaded(self) -> Dict[str, Any]:
        return {
            "enabled": self.__enabled,
            "sources": [r.source_name for r in self.__active_rules()],
        }

    async def try_dispatch(
        self,
        conversation: list[ChatMessage],
        conversation_context: Optional[ConversationContext],
        emit: EmitFn = None,
    ) -> Optional[AsyncGenerator[ChatMessage, None]]:
        if not self.__enabled:
            return None
        for rule in self.__active_rules():
            if rule.matches(conversation_context):
                return await rule.execute(conversation, conversation_context, emit)
        return None

    def __active_rules(self) -> List[DispatchRule]:
        if self.__cached_active is not None:
            return self.__cached_active
        self.__cached_active = [
            rule
            for rule in self.__rules
            if is_source_active(
                rule.source_name,
                rule.enabled,
                self.__include_sources,
                self.__exclude_sources,
            )
        ]
        return self.__cached_active


class DispatchProviderFactory(AgentDependencyFactory):

    @classmethod
    def create(
        cls,
        dispatch_rule_group: DispatchRuleGroup = DispatchRuleGroup(items=[]),
        dispatch_provider_configuration: DispatchProviderConfiguration = (
            DispatchProviderConfiguration()
        ),
    ) -> DispatchProvider:
        return DispatchProvider(
            rules=dispatch_rule_group.items,
            enabled=dispatch_provider_configuration.enabled,
            include_sources=(
                set(dispatch_provider_configuration.include_sources)
                if dispatch_provider_configuration.include_sources
                else None
            ),
            exclude_sources=(
                set(dispatch_provider_configuration.exclude_sources)
                if dispatch_provider_configuration.exclude_sources
                else None
            ),
        )
