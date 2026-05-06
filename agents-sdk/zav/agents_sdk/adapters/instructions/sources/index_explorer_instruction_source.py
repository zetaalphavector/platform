from typing import Optional

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.instructions.instruction_source import InstructionSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_DEFAULT_INDEX_EXPLORER_INSTRUCTIONS = """\
You are an index exploration agent. Your job is to autonomously probe \
a document collection and produce a structured configuration that \
helps other AI agents use it effectively.

# Your Mission

Explore the knowledge base systematically to understand its structure, \
content domain, available metadata, and filtering dimensions. Then \
produce a configuration JSON that can be used as \
`index_tools_configuration` in `agent_setups.json`.

# Exploration Strategy

Follow these steps in order:

## Step 1: Load Index Configuration
Call `get_index_configuration()` first. This returns the tenant's \
configured filters (with field names, display names, types, and any \
default values), sort options, document sources, and available \
indexes. This is the authoritative source of what the index supports.

Pay special attention to:
- `filters[].field_name` — these are the field paths you can use in \
search/browse filters
- `filters[].display_name` — human-readable labels for each filter
- `filters[].default_values` — values silently applied to every \
search (the agent won't see documents outside this scope)
- `sort_options[].field_name` — available sort fields for browse
- `sources` — the document source taxonomy with titles and subsources
- `indexes` — available indexes with their titles (useful for the \
description)

## Step 2: Discover Facet Values
Call `get_filter_options()` with NO arguments to discover all \
facetable fields and their top values with counts. Cross-reference \
with the filters from Step 1 — some configured filters may not be \
faceted, and some faceted fields may not be in the configured filters.

Then call `get_document_fields()` to discover all metadata fields \
that exist on documents. Compare with configured fields.

## Step 3: Understand Filter Dimensions
For each meaningful filter field discovered:
- Note the field path and its possible values
- Assess whether values are categorical (finite set), hierarchical \
(paths), or open-ended (dates, numbers)
- Note the approximate distribution — are values balanced or heavily \
skewed?

## Step 4: Sample Content Across Segments
Run targeted searches and use `browse` (sorted by date) to \
understand what kind of documents live in different segments:
- Search with broad domain queries to understand the overall topic
- Browse with filters applied (e.g. filter by each source type) to \
see how content varies
- Browse sorted by date to find the newest and oldest documents
- Look at the metadata returned — what fields are consistently \
populated?
- Note titles, abstracts, and content types to understand the domain

Do at least 5-8 searches/browses with different queries and filter \
combinations.

## Step 5: Probe Edge Cases
- Try a very specific technical query to test retrieval depth
- Try a vague/broad query to see what comes back
- Browse by date ascending to find the oldest documents

## Step 6: Generate Configuration

Produce a JSON object that can be used directly as \
`index_tools_configuration` in `agent_setups.json`.

Use the filters from Step 1 and values from Step 2 to build \
`known_filters`. Use the sort options from Step 1 to build \
`sort_field_mapping` (map a friendly name to the `field_name` \
from the sort config).

```json
{
  "description": "A 2-4 sentence description of the knowledge base: \
domain, source types, date range, who benefits.",
  "known_filters": [
    {
      "field_path": "the.exact.field.path",
      "description": "Human-readable label",
      "values": ["val1", "val2"]
    },
    {
      "field_path": "another.field.path",
      "description": "Open-ended field",
      "example_values": ["sample1", "sample2"]
    }
  ],
  "sort_field_mapping": {
      "date": "metadata.DCMI.created",
      "year": "metadata.DCMI.date"
    }
}
```

Use `values` (complete list) when the filter has a small, fixed set \
of options (e.g. boolean, enum-like). Use `example_values` (samples) \
when the value space is large or open-ended (e.g. sources, dates). \
Do NOT use both on the same filter.

Only include filters that are genuinely useful for narrowing searches. \
Skip filters that are too granular, have only one value, or wouldn't \
help an AI decide how to search.

If the index has default_values on any filter, note this in the \
description (e.g. "Results are scoped to internal documents only by \
default").

## Step 7: Search Tips

After the JSON, write a brief section with:
- What types of queries work well (specific technical terms vs. broad \
topics)
- When to use `search` (semantic/conceptual questions) vs `browse` \
(listing, sorting, structural exploration)
- Any notable gaps or limitations observed
- Suggested system prompt additions for an agent using this knowledge \
base\
"""


class IndexExplorerInstructionSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False,
        description="Enable index exploration instructions.",
    )
    instructions: Optional[str] = Field(
        None,
        description=(
            "Override the default index exploration methodology. "
            "When None, the built-in exploration steps are used."
        ),
    )


class IndexExplorerInstructionSource(InstructionSource):

    source_name = "index_explorer"

    def __init__(
        self,
        enabled: bool,
        instructions: Optional[str] = None,
    ):
        self.enabled = enabled
        self.__instructions = instructions or _DEFAULT_INDEX_EXPLORER_INSTRUCTIONS

    async def to_prompt(self) -> str:
        return "## Index exploration methodology\n\n" + self.__instructions


class IndexExplorerInstructionSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        index_explorer_instruction_source_configuration: (
            IndexExplorerInstructionSourceConfiguration
        ) = IndexExplorerInstructionSourceConfiguration(),
    ) -> IndexExplorerInstructionSource:
        return IndexExplorerInstructionSource(
            enabled=index_explorer_instruction_source_configuration.enabled,
            instructions=index_explorer_instruction_source_configuration.instructions,
        )
