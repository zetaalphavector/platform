from zav.agents_sdk.adapters.interjections.interjection_source import (
    InterjectionSource,
    InterjectionSourceGroup,
)
from zav.agents_sdk.adapters.interjections.interjections_provider import (
    InterjectionsProvider,
    InterjectionsProviderConfiguration,
    InterjectionsProviderFactory,
)
from zav.agents_sdk.adapters.interjections.sources import (
    AttemptPlateauInterjectionSource,
    AttemptPlateauInterjectionSourceConfiguration,
    AttemptPlateauInterjectionSourceFactory,
    AttemptTranscriptRecorderInterjectionSource,
    AttemptTranscriptRecorderInterjectionSourceConfiguration,
    AttemptTranscriptRecorderInterjectionSourceFactory,
    RefineInterjectionSource,
    RefineInterjectionSourceConfiguration,
    RefineInterjectionSourceFactory,
    RepeatedToolFailureInterjectionSource,
    RepeatedToolFailureInterjectionSourceConfiguration,
    RepeatedToolFailureInterjectionSourceFactory,
    ReviewerInterjectionSource,
    ReviewerInterjectionSourceConfiguration,
    ReviewerInterjectionSourceFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(InterjectionsProviderFactory)
AgentDependencyRegistry.register(AttemptPlateauInterjectionSourceFactory)
AgentDependencyRegistry.register(AttemptTranscriptRecorderInterjectionSourceFactory)
AgentDependencyRegistry.register(RefineInterjectionSourceFactory)
AgentDependencyRegistry.register(RepeatedToolFailureInterjectionSourceFactory)
AgentDependencyRegistry.register(ReviewerInterjectionSourceFactory)

__all__ = [
    "AttemptPlateauInterjectionSource",
    "AttemptPlateauInterjectionSourceConfiguration",
    "AttemptPlateauInterjectionSourceFactory",
    "AttemptTranscriptRecorderInterjectionSource",
    "AttemptTranscriptRecorderInterjectionSourceConfiguration",
    "AttemptTranscriptRecorderInterjectionSourceFactory",
    "InterjectionSource",
    "InterjectionSourceGroup",
    "InterjectionsProvider",
    "InterjectionsProviderConfiguration",
    "InterjectionsProviderFactory",
    "RefineInterjectionSource",
    "RefineInterjectionSourceConfiguration",
    "RefineInterjectionSourceFactory",
    "RepeatedToolFailureInterjectionSource",
    "RepeatedToolFailureInterjectionSourceConfiguration",
    "RepeatedToolFailureInterjectionSourceFactory",
    "ReviewerInterjectionSource",
    "ReviewerInterjectionSourceConfiguration",
    "ReviewerInterjectionSourceFactory",
]
