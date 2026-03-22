from zav.agents_sdk.adapters.skills.disk_skills_source import (
    DiskSkillsSource,
    DiskSkillsSourceConfig,
    DiskSkillsSourceFactory,
)
from zav.agents_sdk.adapters.skills.skill_creation_skills_source import (
    SkillCreationSkillsSource,
    SkillCreationSkillsSourceFactory,
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
    SkillWriteError,
)
from zav.agents_sdk.adapters.skills.tag_note_skills_source import (
    TagNoteSkillsSource,
    TagNoteSkillsSourceConfig,
    TagNoteSkillsSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(SkillsProviderFactory)
AgentDependencyRegistry.register(DiskSkillsSourceFactory)
AgentDependencyRegistry.register(SkillCreationSkillsSourceFactory)
AgentDependencyRegistry.register(TagNoteSkillsSourceFactory)

__all__ = [
    "DiskSkillsSource",
    "DiskSkillsSourceConfig",
    "DiskSkillsSourceFactory",
    "SkillCreationSkillsSource",
    "SkillCreationSkillsSourceFactory",
    "SkillNotFoundError",
    "SkillProperties",
    "SkillReadError",
    "SkillResourceNotFoundError",
    "SkillWriteError",
    "SkillsConfiguration",
    "SkillsProvider",
    "SkillsProviderFactory",
    "SkillsSource",
    "SkillsSourceGroup",
    "TagNoteSkillsSource",
    "TagNoteSkillsSourceConfig",
    "TagNoteSkillsSourceFactory",
]
