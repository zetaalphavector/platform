from typing import List

__all__: List[str] = []

try:
    # Needs the openai and azure-identity packages from the realtime extra.
    from zav.agents_sdk.adapters.realtime_transcription.openai_realtime_transcription import (  # noqa: E501
        OpenAIRealtimeTranscription,
    )
    from zav.agents_sdk.adapters.realtime_transcription.session import (
        RealtimeTranscriptionSession,
    )

    __all__ += ["OpenAIRealtimeTranscription", "RealtimeTranscriptionSession"]
except ImportError:
    pass
