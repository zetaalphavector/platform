import importlib.util

from zav.prompt_completion.client_factories import (
    ChatClientFactory,
    PromptClientFactory,
    PromptWithLogitsClientFactory,
)

if importlib.util.find_spec("anthropic") is not None:
    from zav.prompt_completion.adapters.anthropic_clients import *

if importlib.util.find_spec("openai") is not None:
    from zav.prompt_completion.adapters.openai_clients import *
