from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


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


@dataclass
class RegisteredHit:
    """Both views of a single search result."""

    context_hit: Dict[str, Any]
    search_hit: Dict[str, Any]
    index_type: Optional[str] = None
    index_id: Optional[str] = None


class CitationStore:
    """Accumulates search-result data during tool execution for
    post-processing citation resolution.

    Search tool sources call ``register_hit()`` for each result returned
    to the LLM.  After generation, a ``CitationProcessor`` reads the
    accumulated data to resolve citations, build evidence, and produce
    output ``DocumentContext`` for the next conversational turn.
    """

    def __init__(self) -> None:
        self.__hits: Dict[str, RegisteredHit] = {}
        self.__short_id_map: Dict[str, str] = {}
        self.__doc_id_map: Dict[str, str] = {}

    def register_hit(
        self,
        source_url: str,
        context_hit: Dict[str, Any],
        search_hit: Dict[str, Any],
        short_id: Optional[str] = None,
        index_type: Optional[str] = None,
        index_id: Optional[str] = None,
    ) -> None:
        self.__hits[source_url] = RegisteredHit(
            context_hit=context_hit,
            search_hit=search_hit,
            index_type=index_type,
            index_id=index_id,
        )
        if short_id:
            self.__short_id_map[short_id] = source_url
        doc_id = context_hit.get("document_id") or search_hit.get("id")
        if doc_id:
            self.__doc_id_map[doc_id] = source_url

    def get_hit(self, source_url: str) -> Optional[RegisteredHit]:
        return self.__hits.get(source_url)

    def get_hit_by_document_id(self, document_id: str) -> Optional[RegisteredHit]:
        source_url = self.__doc_id_map.get(document_id)
        if source_url is None:
            return None
        return self.__hits.get(source_url)

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
