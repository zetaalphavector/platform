from typing import Any, Dict, List, Literal, Optional, Set

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import CustomContextItem


class CitationConfiguration(BaseModel):
    """Configures how search results present citation information to the LLM.

    Strategies:
    - ``inline_url``: Results include ``source_url``. LLM cites as
      ``[[N]](source_url "text")``.  Existing evidence pipeline parses
      these directly.
    - ``short_id``: Results include a ``short_id`` (last 8 hex chars of
      the document hash). LLM cites as ``[[short_id]]``.  Post-processing
      (e.g. ``fix_bare_id_citations``) maps IDs back to full URLs.
    - ``deferred``: Results include ``source_url`` but the LLM is
      instructed *not* to cite inline.  A post-processing step matches
      response text to retrieved results and inserts citations.
    """

    strategy: Literal["inline_url", "short_id", "deferred", "sup_numeric"] = Field(
        default="inline_url",
        description="How the agent formats citations in its response.",
    )


class CitationContextContribution(BaseModel):
    document_ids: Set[str] = Field(default_factory=set)
    custom_items: List[CustomContextItem] = Field(default_factory=list)


class RegisteredHit(BaseModel):
    """Both LLM-facing and source-facing views of a citable resource."""

    context_hit: Dict[str, Any]
    search_hit: Dict[str, Any]
    evidence_url: Optional[str] = None
    context_contribution: CitationContextContribution = Field(
        default_factory=CitationContextContribution
    )
    index_type: Optional[str] = None
    index_id: Optional[str] = None


class CitationStore:
    """Accumulates search-result data during tool execution for
    post-processing citation resolution.

    Search tool sources call ``register_hit()`` for each result returned
    to the LLM. After generation, a ``CitationProcessor`` reads the
    accumulated data to resolve citations, build evidence, and produce
    next-turn context.
    """

    def __init__(self) -> None:
        self.__hits: Dict[str, RegisteredHit] = {}
        self.__short_id_map: Dict[str, str] = {}
        self.__doc_id_map: Dict[str, str] = {}

    def register_hit(
        self,
        citation_key: str,
        context_hit: Dict[str, Any],
        search_hit: Dict[str, Any],
        evidence_url: Optional[str] = None,
        context_contribution: Optional[CitationContextContribution] = None,
        short_id: Optional[str] = None,
        index_type: Optional[str] = None,
        index_id: Optional[str] = None,
    ) -> None:
        contribution = context_contribution or CitationContextContribution()
        self.__hits[citation_key] = RegisteredHit(
            context_hit=context_hit,
            search_hit=search_hit,
            evidence_url=evidence_url,
            context_contribution=contribution,
            index_type=index_type,
            index_id=index_id,
        )
        if short_id:
            self.__short_id_map[short_id] = citation_key
        # `context_hit["document_id"]` (internal) and `search_hit["id"]` (federated)
        # are already represented in `contribution.document_ids` / `custom_items`
        # by the registration helpers, so we index off the contribution alone.
        # Precedence on collision: later registrations win (last-write-wins),
        # matching how `__hits` is keyed by citation_key.
        for doc_id in (
            *contribution.document_ids,
            *(item.document_id for item in contribution.custom_items),
        ):
            if doc_id:
                self.__doc_id_map[str(doc_id)] = citation_key

    def get_hit(self, citation_key: str) -> Optional[RegisteredHit]:
        return self.__hits.get(citation_key)

    def get_hit_by_document_id(self, document_id: str) -> Optional[RegisteredHit]:
        citation_key = self.__doc_id_map.get(document_id)
        if citation_key is None:
            return None
        return self.__hits.get(citation_key)

    @property
    def hits(self) -> Dict[str, RegisteredHit]:
        return dict(self.__hits)

    @property
    def short_id_to_source_url(self) -> Dict[str, str]:
        return dict(self.__short_id_map)


class CitationStoreFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(cls) -> CitationStore:
        return CitationStore()
