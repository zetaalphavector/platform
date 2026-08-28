from typing import List, Literal, Set, get_args

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.instructions.instruction_source import InstructionSource
from zav.agents_sdk.adapters.policies.citation import CitationConfiguration
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

CitationStrategy = Literal["inline_url", "short_id", "deferred", "sup_numeric"]
CitableResource = Literal["search_results", "context", "documents", "tags"]

_INLINE_URL_CITATION_INSTRUCTIONS: dict[CitableResource, str] = {
    "search_results": (
        "When citing search results, use the `source_url` field "
        "in this format:\n"
        '[[N]](source_url "relevant text extract")\n'
        "where N is a sequential number starting from 1."
    ),
    "context": (
        "Documents and sources provided to you in the conversation context "
        "(for example, the documents the user is currently viewing) are cited "
        "the same way as search results, using their `source_url` field:\n"
        '[[N]](source_url "relevant text extract")\n'
        "Do not use a plain link; a citation must use the [[N]](...) form to be "
        "shown as a reference."
    ),
    "documents": (
        "When you want to refer to a document, use a markdown link with its "
        "document id in this format:\n"
        "[{document title or a clear anchor text}](/documents/{document ID}_0)."
    ),
    "tags": (
        "When you want to refer to a tag, use a markdown link with its "
        "tag id in this format:\n"
        "[{tag name}](/tags/{tag ID})."
    ),
}

_SHORT_ID_CITATION_INSTRUCTIONS: dict[CitableResource, str] = {
    "search_results": (
        "When citing search results, use the `short_id` field "
        "from each result:\n"
        "[[short_id]]\n"
        "The short_id will be resolved to a full reference automatically."
    ),
}

_DEFERRED_CITATION_INSTRUCTIONS: dict[CitableResource, str] = {
    "search_results": (
        "Focus on providing accurate answers based on the search "
        "results. Do not add inline citations — they will be resolved "
        "automatically based on which search results informed your answer."
    ),
}

_SUP_NUMERIC_CITATION_INSTRUCTIONS: dict[CitableResource, str] = {
    "search_results": (
        "When citing search results, use superscript notation with the "
        "result number: <sup>N</sup> where N matches the result number "
        "from the search results. Place citations inline immediately after "
        "the relevant text."
    ),
}

_STRATEGY_INSTRUCTIONS: dict[CitationStrategy, dict[CitableResource, str]] = {
    "inline_url": _INLINE_URL_CITATION_INSTRUCTIONS,
    "short_id": _SHORT_ID_CITATION_INSTRUCTIONS,
    "deferred": _DEFERRED_CITATION_INSTRUCTIONS,
    "sup_numeric": _SUP_NUMERIC_CITATION_INSTRUCTIONS,
}


def build_citation_instructions(
    citation_config: CitationConfiguration,
    citable_resources: Set[CitableResource],
) -> str:
    resource_map = _STRATEGY_INSTRUCTIONS.get(
        citation_config.strategy, _INLINE_URL_CITATION_INSTRUCTIONS
    )
    return "\n\n".join(
        instruction
        for resource in get_args(CitableResource)
        if resource in citable_resources
        and (instruction := resource_map.get(resource)) is not None
    )


class CitationInstructionSourceConfiguration(BaseModel):
    enabled: bool = Field(
        False, description="Include citation format instructions in the system prompt."
    )
    citable_resources: List[CitableResource] = Field(
        default_factory=lambda: ["search_results"],
        description=(
            "Which resource types to include citation instructions for. "
            "'search_results' is always implied by the strategy. "
            "Add 'documents' and/or 'tags' to include linking instructions."
        ),
    )


class CitationInstructionSource(InstructionSource):

    source_name = "citation_instructions"

    def __init__(
        self,
        citation_config: CitationConfiguration,
        enabled: bool,
        citable_resources: Set[CitableResource],
    ):
        self.enabled = enabled
        self.__citation_config = citation_config
        self.__citable_resources = citable_resources

    async def to_prompt(self) -> str:
        return (
            "## Citations\n\n"
            "When using any search results to inform your answer, "
            "cite them using the following format:\n\n"
            + build_citation_instructions(
                self.__citation_config, self.__citable_resources
            )
        )


class CitationInstructionSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        citation_configuration: CitationConfiguration = CitationConfiguration(),
        citation_instruction_source_configuration: CitationInstructionSourceConfiguration = CitationInstructionSourceConfiguration(),  # noqa: E501
    ) -> CitationInstructionSource:
        return CitationInstructionSource(
            citation_config=citation_configuration,
            enabled=citation_instruction_source_configuration.enabled,
            citable_resources=set(
                citation_instruction_source_configuration.citable_resources
            ),
        )
