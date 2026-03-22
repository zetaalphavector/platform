from typing import ClassVar, Dict, List

from zav.agents_sdk.adapters.search.citation import (
    CitationConfig,
    build_citation_instructions,
)
from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillResourceNotFoundError,
    SkillsSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_SKILL_NAME = "search"

_SEARCH_SKILL_DESCRIPTION = (
    "Instructions for searching the knowledge base, "
    "using filters, and citing results."
)

_SEARCH_INSTRUCTIONS_BODY = """\
## Search Tools

You have access to search tools for finding documents \
in the knowledge base.

### `search`
Use this tool to search for documents. You can provide:
- A natural language `query`
- Optional `filters` as key-value pairs \
(e.g. `{"source": "arxiv", "year": "2024"}`)
  - Single values are matched exactly
  - Lists match any value \
(e.g. `{"source": ["arxiv", "pubmed"]}`)
- Use `get_filter_options` first to discover available \
filter fields before applying filters you haven't seen.

### `get_filter_options`
Use this tool to discover what filter fields are available \
and their possible values.
- Call without arguments to see all available filter fields.
- Provide `field_paths` to get values for specific fields.
- Provide a `query` to scope the facet counts.\
"""


def _build_search_skill_body(citation_config: CitationConfig) -> str:
    body = _SEARCH_INSTRUCTIONS_BODY
    body += "\n\n### Citations\n"
    body += build_citation_instructions(citation_config)
    return body


class SearchSkillsSource(SkillsSource):
    """Provides search instructions as a discoverable skill.

    This is the skills-paradigm counterpart of ``IndexToolsSource``.
    While ``IndexToolsSource`` provides the callable ``search`` and
    ``get_filter_options`` tools, this source provides the instructions
    for how to use them effectively — discoverable via ``use_skill``.

    Citation instructions vary based on the configured
    ``CitationConfig.strategy``.
    """

    source_name: ClassVar[str] = "search_skills"

    def __init__(
        self,
        citation_config: CitationConfig,
    ):
        self.__citation_config = citation_config

    async def discover(self) -> Dict[str, SkillProperties]:
        return {
            _SKILL_NAME: SkillProperties(
                name=_SKILL_NAME,
                description=_SEARCH_SKILL_DESCRIPTION,
            ),
        }

    async def get_skill_body(self, skill_name: str) -> str:
        if skill_name != _SKILL_NAME:
            available = await self.discover()
            raise SkillNotFoundError(skill_name, list(available.keys()))
        return _build_search_skill_body(self.__citation_config)

    async def get_resources(self, skill_name: str) -> List[str]:
        if skill_name != _SKILL_NAME:
            available = await self.discover()
            raise SkillNotFoundError(skill_name, list(available.keys()))
        return []

    async def read_resource(self, skill_name: str, path: str) -> str:
        if skill_name != _SKILL_NAME:
            available = await self.discover()
            raise SkillNotFoundError(skill_name, list(available.keys()))
        available_resources = await self.get_resources(skill_name)
        raise SkillResourceNotFoundError(skill_name, path, available_resources)


class SearchSkillsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        citation_configuration: CitationConfig = CitationConfig(),
    ) -> SearchSkillsSource:
        return SearchSkillsSource(
            citation_config=citation_configuration,
        )
