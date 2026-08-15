from zav.api import AppResource

from zav.agents_sdk.adapters.stream_buffer import StreamBufferRegistry

get_stream_buffer_registry = AppResource[StreamBufferRegistry]("stream_buffer_registry")
