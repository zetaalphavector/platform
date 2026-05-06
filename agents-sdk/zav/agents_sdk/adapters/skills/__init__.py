from zav.agents_sdk.adapters.skills.skills_provider import (
    SkillsProvider,
    SkillsProviderConfiguration,
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
from zav.agents_sdk.adapters.skills.sources.disk_skills_source import (
    DiskSkillsSource,
    DiskSkillsSourceConfiguration,
    DiskSkillsSourceFactory,
)
from zav.agents_sdk.adapters.skills.sources.image_search_skills_source import (
    ImageSearchSkillsSource,
    ImageSearchSkillsSourceConfiguration,
    ImageSearchSkillsSourceFactory,
)
from zav.agents_sdk.adapters.skills.sources.in_memory_skills_source import (
    InMemorySkillDefinition,
    InMemorySkillsSource,
    InMemorySkillsSourceConfiguration,
    InMemorySkillsSourceFactory,
)
from zav.agents_sdk.adapters.skills.sources.platform_docs_skills_source import (
    PlatformDocsSkillsSource,
    PlatformDocsSkillsSourceConfiguration,
    PlatformDocsSkillsSourceFactory,
)
from zav.agents_sdk.adapters.skills.sources.skill_creation_skills_source import (
    SkillCreationSkillsSource,
    SkillCreationSkillsSourceConfiguration,
    SkillCreationSkillsSourceFactory,
)
from zav.agents_sdk.adapters.skills.sources.skill_note_skills_source import (
    SkillNoteSkillsSource,
    SkillNoteSkillsSourceConfiguration,
    SkillNoteSkillsSourceFactory,
)
from zav.agents_sdk.adapters.skills.sources.tag_note_skills_source import (
    TagNoteSkillsSource,
    TagNoteSkillsSourceConfiguration,
    TagNoteSkillsSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(SkillsProviderFactory)
AgentDependencyRegistry.register(DiskSkillsSourceFactory)
AgentDependencyRegistry.register(ImageSearchSkillsSourceFactory)
AgentDependencyRegistry.register(InMemorySkillsSourceFactory)
AgentDependencyRegistry.register(SkillCreationSkillsSourceFactory)
AgentDependencyRegistry.register(TagNoteSkillsSourceFactory)
AgentDependencyRegistry.register(SkillNoteSkillsSourceFactory)
AgentDependencyRegistry.register(PlatformDocsSkillsSourceFactory)

__all__ = [
    "DiskSkillsSource",
    "DiskSkillsSourceConfiguration",
    "DiskSkillsSourceFactory",
    "ImageSearchSkillsSource",
    "ImageSearchSkillsSourceConfiguration",
    "ImageSearchSkillsSourceFactory",
    "InMemorySkillDefinition",
    "InMemorySkillsSource",
    "InMemorySkillsSourceConfiguration",
    "InMemorySkillsSourceFactory",
    "SkillCreationSkillsSource",
    "SkillCreationSkillsSourceConfiguration",
    "SkillCreationSkillsSourceFactory",
    "SkillNotFoundError",
    "SkillProperties",
    "SkillReadError",
    "SkillResourceNotFoundError",
    "SkillWriteError",
    "SkillsProviderConfiguration",
    "SkillsProvider",
    "SkillsProviderFactory",
    "SkillsSource",
    "SkillsSourceGroup",
    "TagNoteSkillsSource",
    "TagNoteSkillsSourceConfiguration",
    "TagNoteSkillsSourceFactory",
    "SkillNoteSkillsSource",
    "SkillNoteSkillsSourceConfiguration",
    "SkillNoteSkillsSourceFactory",
    "PlatformDocsSkillsSource",
    "PlatformDocsSkillsSourceConfiguration",
    "PlatformDocsSkillsSourceFactory",
]
