from typing import Literal

from zav.pydantic_compat import BaseModel, Field


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
