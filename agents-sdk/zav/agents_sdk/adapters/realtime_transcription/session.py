import asyncio
import json
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, Optional, Union

from openai.resources.realtime.realtime import AsyncRealtimeConnection
from websockets.exceptions import ConnectionClosed
from zav.logging import logger

from zav.agents_sdk.domain.realtime_transcription import error_event

# Decoded input audio per second, by audio.input.format.type: 24 kHz PCM16 (the
# default) and 8 kHz G.711.
_BYTES_PER_SECOND = {"audio/pcm": 48_000, "audio/pcmu": 8_000, "audio/pcma": 8_000}
# How far audio may run ahead of the session clock. A microphone streams in real
# time; the slack absorbs audio a frontend buffers before the socket opens.
_AUDIO_SLACK_SECONDS = 10


class _AudioLimitReached(Exception):
    pass


def _child(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    child = parent.get(key)
    if not isinstance(child, dict):
        child = parent[key] = {}
    return child


def session_update(
    model: str,
    language_hint: Dict[str, Any],
    client_update: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # The client's session.update as sent, except the session type and model,
    # which are the tenant's. A client's own language hint wins over the
    # configured one; models reject language and languages together.
    update = dict(client_update or {"type": "session.update"})
    session = _child(update, "session")
    session["type"] = "transcription"
    transcription = _child(_child(_child(session, "audio"), "input"), "transcription")
    transcription["model"] = model
    if "language" not in transcription and "languages" not in transcription:
        transcription.update(language_hint)
    return update


def _bytes_per_second(client_update: Dict[str, Any], current: int) -> int:
    node: Any = client_update
    for key in ("session", "audio", "input", "format", "type"):
        if not isinstance(node, dict):
            return current
        node = node.get(key)
    return _BYTES_PER_SECOND.get(str(node), current)


def _decoded_length(audio: Any) -> int:
    # The size of the base64 payload once decoded, without decoding it.
    if not isinstance(audio, str):
        return 0
    return len(audio) * 3 // 4 - audio.count("=", len(audio) - 2)


def _json_object(frame: Union[str, bytes]) -> Optional[Dict[str, Any]]:
    try:
        event = json.loads(frame)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


class RealtimeTranscriptionSession:
    """Relays the OpenAI Realtime transcription protocol between a client and
    the provider. Only session.update is touched, to pin the tenant's model.
    Ends at max_seconds of wall-clock time, or when the audio sent runs more than
    the slack ahead of the session clock."""

    def __init__(
        self,
        connection: AsyncRealtimeConnection,
        max_seconds: float,
        model: str,
        language_hint: Dict[str, Any],
        clock: Optional[Callable[[], float]] = None,
    ):
        self.__connection = connection
        self.__max_seconds = max_seconds
        self.__model = model
        self.__language_hint = language_hint
        self.__clock = clock or asyncio.get_running_loop().time

    async def run(
        self, browser_frames: AsyncIterator[Union[str, bytes]]
    ) -> AsyncGenerator[str, None]:
        started = self.__clock()
        deadline = started + self.__max_seconds
        upstream = asyncio.create_task(self.__forward(browser_frames, started))
        receive: Optional["asyncio.Task[bytes]"] = None
        try:
            while True:
                receive = receive or asyncio.create_task(self.__connection.recv_bytes())
                done, _ = await asyncio.wait(
                    {upstream, receive},
                    timeout=max(deadline - self.__clock(), 0),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # A frame already buffered when the deadline passes still
                # completes within a zero timeout; it must not extend the session.
                if receive in done and self.__clock() < deadline:
                    try:
                        frame = receive.result()
                    except ConnectionClosed:
                        return
                    receive = None
                    yield frame.decode()
                elif upstream in done:
                    try:
                        upstream.result()
                    except _AudioLimitReached:
                        yield self.__audio_limit_error()
                    return
                else:
                    yield self.__time_limit_error()
                    return
        finally:
            for task in (upstream, receive):
                if task is not None:
                    task.cancel()

    async def __forward(
        self, browser_frames: AsyncIterator[Union[str, bytes]], started: float
    ) -> None:
        # Providers bill audio, and a client can send it faster than real time.
        # Holding audio to the session clock plus the slack keeps a session, and
        # a reconnect loop, to about real time.
        bytes_per_second = _BYTES_PER_SECOND["audio/pcm"]
        audio_seconds = 0.0
        dropped_non_json = False
        async for frame in browser_frames:
            event = _json_object(frame)
            if event is None:
                # Providers accept only JSON events. Logged once: a client
                # streaming binary audio sends dozens of frames a second.
                if not dropped_non_json:
                    logger.info("Realtime transcription dropped non-JSON client frames")
                    dropped_non_json = True
                continue
            if event.get("type") == "session.update":
                bytes_per_second = _bytes_per_second(event, bytes_per_second)
                event = session_update(self.__model, self.__language_hint, event)
            elif event.get("type") == "input_audio_buffer.append":
                audio_seconds += _decoded_length(event.get("audio")) / bytes_per_second
                if audio_seconds > self.__clock() - started + _AUDIO_SLACK_SECONDS:
                    raise _AudioLimitReached()
            try:
                await self.__connection.send(event)  # type: ignore[arg-type]
            except ConnectionClosed:
                # The provider ended the session; run() sees the same close.
                return

    @staticmethod
    def __audio_limit_error() -> str:
        logger.info("Dictation session closed: audio limit reached")
        return error_event(
            "dictation_limit_reached", "audio_limit", "Dictation audio limit reached"
        )

    @staticmethod
    def __time_limit_error() -> str:
        logger.info("Dictation session closed: time limit reached")
        return error_event(
            "dictation_limit_reached", "time_limit", "Dictation time limit reached"
        )
