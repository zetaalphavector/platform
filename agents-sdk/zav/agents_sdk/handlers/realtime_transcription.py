import time
from typing import AsyncGenerator, List, Optional

from zav.logging import logger
from zav.message_bus import Message, StreamCommandHandlerRegistry

from zav.agents_sdk.adapters.realtime_transcription import OpenAIRealtimeTranscription
from zav.agents_sdk.domain.llm_client_factory import LLMClientFactory, LLMNotConfigured
from zav.agents_sdk.domain.llm_configuration_store import LLMConfigurationStoreFactory
from zav.agents_sdk.domain.realtime_transcription import (
    RealtimeTranscriptionNotConfigured,
    RealtimeTranscriptionSetupRetriever,
)
from zav.agents_sdk.handlers import commands


@StreamCommandHandlerRegistry.register(commands.StreamRealtimeTranscription)
async def handle_stream_realtime_transcription(
    cmd: commands.StreamRealtimeTranscription,
    queue: List[Message],
    realtime_transcription_setup_retriever: Optional[
        RealtimeTranscriptionSetupRetriever
    ] = None,
    llm_configuration_store_factory: Optional[LLMConfigurationStoreFactory] = None,
) -> AsyncGenerator[str, None]:
    if (
        realtime_transcription_setup_retriever is None
        or llm_configuration_store_factory is None
    ):
        raise RealtimeTranscriptionNotConfigured(
            "This host serves no realtime transcription."
        )
    setup = await realtime_transcription_setup_retriever.get(cmd.tenant)
    if setup is None:
        raise RealtimeTranscriptionNotConfigured(
            f"Realtime transcription is not set up for {cmd.tenant}."
        )
    store = await llm_configuration_store_factory.create(cmd.tenant)
    try:
        transcription = LLMClientFactory(
            llm_configuration_store=store,
            default_llm_configuration_name=setup.llm_configuration_name,
        ).create_client(OpenAIRealtimeTranscription.from_llm_configuration)
    except (LLMNotConfigured, ValueError) as error:
        raise RealtimeTranscriptionNotConfigured(str(error)) from error

    started = time.monotonic()
    try:
        async with transcription.open_session() as session:
            frames = session.run(cmd.browser_frames)
            try:
                async for frame in frames:
                    yield frame
            finally:
                await frames.aclose()
    finally:
        logger.info(
            f"Realtime transcription session ended: tenant={cmd.tenant} "
            f"llm_configuration={setup.llm_configuration_name} "
            f"user={cmd.requester_uuid} "
            f"seconds={time.monotonic() - started:.1f}"
        )
