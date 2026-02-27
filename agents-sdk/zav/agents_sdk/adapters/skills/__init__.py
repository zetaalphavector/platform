from zav.agents_sdk.adapters.skills.disk_skills_source import (
    DiskSkillsSource,
    DiskSkillsSourceConfig,
    DiskSkillsSourceFactory,
)
from zav.agents_sdk.adapters.skills.skills_provider import (
    SkillsConfiguration,
    SkillsProvider,
    SkillsProviderFactory,
)
from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillReadError,
    SkillResourceNotFoundError,
    SkillsSource,
    SkillsSourceGroup,
)
from zav.agents_sdk.adapters.skills.tag_note_skills_source import (
    TagNoteSkillsSource,
    TagNoteSkillsSourceConfig,
    TagNoteSkillsSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(SkillsProviderFactory)
AgentDependencyRegistry.register(DiskSkillsSourceFactory)
AgentDependencyRegistry.register(TagNoteSkillsSourceFactory)

__all__ = [
    "DiskSkillsSource",
    "DiskSkillsSourceConfig",
    "DiskSkillsSourceFactory",
    "SkillNotFoundError",
    "SkillProperties",
    "SkillReadError",
    "SkillResourceNotFoundError",
    "SkillsConfiguration",
    "SkillsProvider",
    "SkillsProviderFactory",
    "SkillsSource",
    "SkillsSourceGroup",
    "TagNoteSkillsSource",
    "TagNoteSkillsSourceConfig",
    "TagNoteSkillsSourceFactory",
]
