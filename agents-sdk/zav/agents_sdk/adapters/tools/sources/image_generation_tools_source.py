import base64
import re
from typing import Any, Dict, List, Literal, Optional

from zav.logging import logger
from zav.prompt_completion import (
    ContentFilterError,
    ImageGenerationClient,
    ImageGenerationRequest,
    RateLimitError,
)
from zav.pydantic_compat import Field

from zav.agents_sdk.adapters.agent_state.conversation_image_store import (
    ConversationImageStore,
)
from zav.agents_sdk.adapters.tools.tools_source import (
    ToolsSource,
    ToolsSourceConfiguration,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, exclude_fields, streamable

_ASPECT_RATIO = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")

_SYSTEM_PROMPT_SECTION = """\
## Image generation

You can create pictures for the user with the `generate_image` tool: \
illustrations, diagrams, mock-ups, product shots, and other visuals.

- Write a specific prompt: subject, setting, style, composition, and any text \
that must appear in the image.
- Pick the `aspect_ratio` that fits the use: `1:1` by default, `16:9` for a \
slide, `3:1` for a banner, `2:3` for a poster. The result reports the size \
actually rendered.
- `generate_image` returns an `image_id`; the picture itself is shown to the \
user, you do not see it. Embed each image inline as a Markdown image where it \
belongs in your answer, using the id as the source: \
`![<short caption describing the image>](image:<image_id>)`.
- When the result is `blocked`, the provider's safety system rejected the \
prompt or the picture. Do not retry the same prompt: change the scene or tell \
the user it cannot be generated.
- When the result is `rate_limited`, do not retry now; tell the user to try \
again later.
- At most {max_images} image generations per answer, failed attempts included.\
"""

_COMPLETED_TEXT = (
    "{% if status == 'ok' %}Generated image."
    "{% elif error == 'rate_limited' %}Image generation is rate limited."
    "{% elif error == 'blocked' %}"
    "Image generation was blocked by the provider's safety system."
    "{% elif error == 'limit_reached' %}Image limit reached."
    "{% else %}Image generation failed.{% endif %}"
)


class ImageGenerationToolsSourceConfiguration(ToolsSourceConfiguration):
    """The image model is named by the agent's `image_llm_selection_configuration`."""

    enabled: bool = False
    max_images_per_turn: int = Field(
        default=2,
        ge=1,
        le=10,
        description=(
            "Image generations the agent may attempt in one answer, failed "
            "attempts included."
        ),
    )
    request_timeout_seconds: float = Field(
        default=180.0,
        gt=0,
        le=600,
        description="Give up on a generation that does not finish within this time.",
    )


class ImageGenerationToolsSource(ToolsSource):

    source_name = "image_generation_tools"
    source_description = "Generate images from a text prompt."

    def __init__(
        self,
        image_generation_tools_source_configuration: (
            ImageGenerationToolsSourceConfiguration
        ),
        image_generation_client: ImageGenerationClient,
        conversation_image_store: ConversationImageStore,
    ):
        self.__config = image_generation_tools_source_configuration
        self.defer_tools = image_generation_tools_source_configuration.defer_tools
        self.enabled = image_generation_tools_source_configuration.enabled
        self.__client = image_generation_client
        self.__image_store = conversation_image_store
        self.__attempts = 0

    async def get_tools(self) -> List[Tool]:
        return [
            Tool.from_callable(
                name="generate_image",
                executable=self.generate_image,
            ),
        ]

    async def to_prompt(self) -> str:
        if not self.enabled:
            return ""
        return _SYSTEM_PROMPT_SECTION.format(
            max_images=self.__config.max_images_per_turn
        )

    @streamable(
        running_text="Generating image...",
        completed_text=_COMPLETED_TEXT,
        llm_response_transform=exclude_fields("image_uri"),
    )
    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        background: Optional[Literal["transparent", "opaque"]] = None,
    ) -> Dict[str, Any]:
        """Generate an image from a text prompt and show it to the user.

        Args:
            prompt: What the image should show: subject, setting, style,
                composition, and any text that must appear in it.
            aspect_ratio: Width to height as `W:H`, e.g. `16:9`.
            background: `transparent` for logos, icons and cut-outs, `opaque`
                to force a filled background; unset lets the model decide.

        Returns:
            On success, a dict with ``image_id``, the rendered ``size`` and a
            short ``description``.
            Embed the image inline as a Markdown image where it belongs:
            ``![<caption>](image:<image_id>)``. On failure, an error dict whose
            ``error`` is ``blocked`` (safety system, do not retry the same
            prompt), ``rate_limited`` (do not retry now), ``limit_reached`` or
            ``failed``.
        """
        ratio = _ASPECT_RATIO.match(aspect_ratio)
        if ratio is None or min(int(ratio[1]), int(ratio[2])) < 1:
            return {
                "status": "error",
                "error": "failed",
                "message": (
                    f"aspect_ratio {aspect_ratio!r} is not `W:H` with positive "
                    "integers, e.g. `16:9`."
                ),
            }
        # Concurrent tool calls: claim the slot before the await; failures keep it.
        if self.__attempts >= self.__config.max_images_per_turn:
            return {
                "status": "error",
                "error": "limit_reached",
                "message": (
                    f"At most {self.__config.max_images_per_turn} image "
                    "generations per answer; use the ones already generated."
                ),
            }
        self.__attempts += 1
        request = ImageGenerationRequest(
            prompt=prompt,
            aspect_ratio=(int(ratio[1]), int(ratio[2])),
            background=background,
            timeout_seconds=self.__config.request_timeout_seconds,
        )
        try:
            result = await self.__client.generate(request)
        except RateLimitError as e:
            retry_hint = (
                f" in {int(e.retry_after_seconds)} seconds"
                if e.retry_after_seconds
                else " later"
            )
            return {
                "status": "error",
                "error": "rate_limited",
                "retry_after_seconds": e.retry_after_seconds,
                "message": (
                    "The image model is rate limited. Do not retry now; tell the "
                    f"user to try again{retry_hint}."
                ),
            }
        except ContentFilterError as e:
            return {
                "status": "error",
                "error": "blocked",
                "message": (
                    "The provider's safety system rejected this request: "
                    f"{e}. Do not retry the same prompt: change the scene or "
                    "tell the user it cannot be generated."
                ),
            }
        except Exception as e:
            logger.warning(f"Image generation failed: {e}")
            return {"status": "error", "error": "failed", "message": str(e)}

        encoded = base64.b64encode(result.image_bytes).decode("ascii")
        image_uri = f"data:{result.mime_type};base64,{encoded}"
        image_id = self.__image_store.register_image(image_uri)
        response: Dict[str, Any] = {
            "status": "ok",
            "image_id": image_id,
            "size": f"{result.width}x{result.height}",
            "description": f"Generated image: {prompt[:80]}",
        }
        if result.revised_prompt:
            response["revised_prompt"] = result.revised_prompt
        response["image_uri"] = image_uri
        return response


class ImageGenerationToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        conversation_image_store: ConversationImageStore,
        image_generation_client: ImageGenerationClient,
        image_generation_tools_source_configuration: (
            ImageGenerationToolsSourceConfiguration
        ) = ImageGenerationToolsSourceConfiguration(),
    ) -> ImageGenerationToolsSource:
        return ImageGenerationToolsSource(
            image_generation_tools_source_configuration=(
                image_generation_tools_source_configuration
            ),
            image_generation_client=image_generation_client,
            conversation_image_store=conversation_image_store,
        )
