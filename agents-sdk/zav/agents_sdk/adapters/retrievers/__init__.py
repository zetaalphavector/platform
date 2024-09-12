import importlib.util

from zav.agents_sdk.adapters.retrievers.zav_retriever import (
    ZAVRetriever,
    ZAVRetrieverFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(ZAVRetrieverFactory)


__all__ = [
    "AgentDependencyRegistry",
    "ZAVRetriever",
    "ZAVRetrieverFactory",
]

if importlib.util.find_spec("langchain_core") is not None:
    from zav.agents_sdk.adapters.retrievers.zav_langchain_retriever import (
        ZAVLangchainRetriever,
        ZAVLangchainStore,
        ZAVLangchainStoreFactory,
    )

    AgentDependencyRegistry.register(ZAVLangchainStoreFactory)
    __all__ += [
        "ZAVLangchainRetriever",
        "ZAVLangchainStore",
        "ZAVLangchainStoreFactory",
    ]
