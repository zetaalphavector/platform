import importlib.util

if importlib.util.find_spec("anthropic") is not None:
    from zav.prompt_completion.adapters.anthropic_clients import *

if importlib.util.find_spec("openai") is not None:
    from zav.prompt_completion.adapters.openai_clients import *
