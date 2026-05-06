from typing import ClassVar, Dict, List

from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.skills.skills_source import (
    SkillNotFoundError,
    SkillProperties,
    SkillResourceNotFoundError,
    SkillsSource,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_SKILL_NAME = "image_search"

_SKILL_DESCRIPTION = (
    "Instructions for searching the knowledge base using images "
    "from the conversation and validating results visually."
)

_DEFAULT_IMAGE_SEARCH_BODY = """\
## Image Search

When the user shares an image and asks you to find related documents, \
follow these steps:

### 1. Describe the image
Look at the image carefully and formulate a detailed search query \
that captures its content. Include:
- Key visual elements (shapes, diagrams, charts, text, labels)
- Technical details (numbers, dimensions, colors, patterns)
- The type of content (photograph, diagram, chart, screenshot, etc.)

### 2. Search
Call the `search` tool with your description as the query. \
Do not mention the image ID or internal processing steps to \
the user.

### 3. Validate results
For each promising result that has a `document_id` and page \
metadata, call `validate_image_hit` to get a visual comparison:
- Pass the `image_id` from the conversation context
- Pass the `document_id` from the search result (base ID without \
chunk suffix, e.g. if the result ID is `abc123_4`, use `abc123`)
- Pass the `page_number` from the result metadata if available, \
otherwise use page 1
- Optionally pass the `document_title` for better context

The tool returns an `analysis` field describing how the document \
page relates to the user's image. Use this analysis to decide \
which results are relevant and which to discard.

### 4. Present results
Present the validated search results to the user, explaining \
how they relate to the image content.\
"""


class ImageSearchSkillsSourceConfiguration(BaseModel):
    enabled: bool = Field(False, description="Enable image search skills source.")


class ImageSearchSkillsSource(SkillsSource):

    source_name: ClassVar[str] = "image_search_skills"

    def __init__(
        self,
        image_search_skill_body: str,
        enabled: bool,
    ):
        self.enabled = enabled
        self.__body = image_search_skill_body

    async def discover(self) -> Dict[str, SkillProperties]:
        return {
            _SKILL_NAME: SkillProperties(
                name=_SKILL_NAME,
                description=_SKILL_DESCRIPTION,
            ),
        }

    async def get_skill_body(self, skill_name: str) -> str:
        if skill_name != _SKILL_NAME:
            available = await self.discover()
            raise SkillNotFoundError(skill_name, list(available.keys()))
        return self.__body

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


class ImageSearchSkillsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        image_search_skill_body: str = _DEFAULT_IMAGE_SEARCH_BODY,
        image_search_skills_source_configuration: ImageSearchSkillsSourceConfiguration = ImageSearchSkillsSourceConfiguration(),  # noqa: E501
    ) -> ImageSearchSkillsSource:
        return ImageSearchSkillsSource(
            image_search_skill_body=image_search_skill_body,
            enabled=image_search_skills_source_configuration.enabled,
        )
