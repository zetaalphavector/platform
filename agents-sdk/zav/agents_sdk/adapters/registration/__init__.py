from zav.agents_sdk.adapters.registration.registration_service import (
    RegistrationService,
    RegistrationServiceFactory,
    UserInfo,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry

AgentDependencyRegistry.register(RegistrationServiceFactory)

__all__ = [
    "RegistrationService",
    "RegistrationServiceFactory",
    "UserInfo",
]
