from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from zav.pydantic_compat import BaseModel

from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory


class CitationConfig(BaseModel):
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

    strategy: Literal["inline_url", "short_id", "deferred"] = "inline_url"


_INLINE_URL_CITATION_INSTRUCTIONS = (
    "When citing search results, use the `source_url` field "
    "in this format:\n"
    '[[N]](source_url "relevant text extract")\n'
    "where N is a sequential number starting from 1."
)

_SHORT_ID_CITATION_INSTRUCTIONS = (
    "When citing search results, use the `short_id` field "
    "from each result:\n"
    "[[short_id]]\n"
    "The short_id will be resolved to a full reference automatically."
)

_DEFERRED_CITATION_INSTRUCTIONS = (
    "Focus on providing accurate answers based on the search "
    "results. Do not add inline citations — they will be resolved "
    "automatically based on which search results informed your answer."
)


def build_citation_instructions(citation_config: CitationConfig) -> str:
    if citation_config.strategy == "short_id":
        return _SHORT_ID_CITATION_INSTRUCTIONS
    elif citation_config.strategy == "deferred":
        return _DEFERRED_CITATION_INSTRUCTIONS
    return _INLINE_URL_CITATION_INSTRUCTIONS


@dataclass
class RegisteredHit:
    """Both views of a single search result."""

    context_hit: Dict[str, Any]
    search_hit: Dict[str, Any]


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

    def register_hit(
        self,
        source_url: str,
        context_hit: Dict[str, Any],
        search_hit: Dict[str, Any],
        short_id: Optional[str] = None,
    ) -> None:
        self.__hits[source_url] = RegisteredHit(
            context_hit=context_hit, search_hit=search_hit
        )
        if short_id:
            self.__short_id_map[short_id] = source_url

    def get_hit(self, source_url: str) -> Optional[RegisteredHit]:
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
