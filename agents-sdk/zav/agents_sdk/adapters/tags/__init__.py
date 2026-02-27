from zav.agents_sdk.adapters.tags.tags_service import TagsService, TagsServiceFactory
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(TagsServiceFactory)

__all__ = [
    "TagsService",
    "TagsServiceFactory",
]
