from zav.agents_sdk.adapters.user_documents.user_documents_service import (
    UserDocument,
    UserDocumentAccessRight,
    UserDocumentCreateRequest,
    UserDocumentMetadata,
    UserDocumentResource,
    UserDocumentSharing,
    UserDocumentsService,
    UserDocumentsServiceFactory,
    UserDocumentStatus,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(UserDocumentsServiceFactory)

__all__ = [
    "UserDocument",
    "UserDocumentAccessRight",
    "UserDocumentCreateRequest",
    "UserDocumentMetadata",
    "UserDocumentResource",
    "UserDocumentSharing",
    "UserDocumentsService",
    "UserDocumentsServiceFactory",
    "UserDocumentStatus",
]
