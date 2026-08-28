from typing import Callable, Optional, Sequence, TypeVar

from zav.llm_domain import LLMClientConfiguration
from zav.llm_tracing import Span
from zav.logging import logger
from zav.prompt_completion import ChatClientFactory, ChatCompletionClient

from zav.agents_sdk.domain.llm_configuration_store import LLMConfigurationStore

T = TypeVar("T")


class LLMNotConfigured(Exception):
    pass


class LLMClientFactory:
    """Resolves LLM selections by name and builds clients; the configurations
    (and their credentials) never leave it. Built by the DependencyResolver per
    walk from the setup's names and injected by parameter type.

    A requested name (runtime input via the model policy) is honored when
    ``allowed_llm_configuration_names`` is None (unrestricted) or contains it,
    and falls back to the default otherwise."""

    def __init__(
        self,
        llm_configuration_store: Optional[LLMConfigurationStore] = None,
        default_llm_configuration_name: Optional[str] = None,
        allowed_llm_configuration_names: Optional[Sequence[str]] = None,
    ):
        self.__llm_configuration_store = llm_configuration_store
        self.__default_name = default_llm_configuration_name
        self.__allowed = (
            None
            if allowed_llm_configuration_names is None
            else tuple(allowed_llm_configuration_names)
        )

    def resolve_selection(self, requested: Optional[str] = None) -> Optional[str]:
        if requested and isinstance(requested, str):
            if (
                self.__allowed is None
                or requested == self.__default_name
                or requested in self.__allowed
            ):
                return requested
            logger.warning(
                "Requested llm_configuration_name '%s' is not allowed; "
                "using the default.",
                requested,
            )
        return self.__default_name

    def __client_configuration(
        self, name: Optional[str]
    ) -> Optional[LLMClientConfiguration]:
        if name is None:
            return None
        if self.__llm_configuration_store is None:
            logger.warning(
                "llm_configuration_name '%s' is selected but no LLM "
                "configuration store is available.",
                name,
            )
            return None
        configuration = self.__llm_configuration_store.get(name)
        if configuration is None:
            logger.warning("Unknown llm_configuration_name '%s'.", name)
        return configuration

    def create_client(
        self,
        builder: Callable[[LLMClientConfiguration], T],
        requested: Optional[str] = None,
    ) -> T:
        # The bridge for the SDK's client factories: the configuration goes to
        # the builder and is never returned.
        name = self.resolve_selection(requested=requested)
        configuration = self.__client_configuration(name)
        if configuration is None:
            raise LLMNotConfigured(
                f"No LLM configuration in scope (selection: {name!r})."
            )
        return builder(configuration)

    def create_chat_completion_client(
        self,
        requested: Optional[str] = None,
        span: Optional[Span] = None,
    ) -> ChatCompletionClient:
        return self.create_client(
            lambda configuration: ChatClientFactory.create(configuration, span=span),
            requested=requested,
        )
