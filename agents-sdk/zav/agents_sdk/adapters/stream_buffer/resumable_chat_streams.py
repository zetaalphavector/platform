"""Resumable chat streams: producing a turn, re-attaching to it, and following
it across pods.

The single story the pieces here share — stated once, referenced by the rest:

* **One overwritten snapshot per conversation.** The durable recording is not
  an append log: it is the latest cumulative ``ChatMessage`` of whichever turn
  occupies the conversation's slot, plus the producer's yield ``index`` at that
  point. Each streamed message already holds the whole answer so far, so
  keeping only the newest is loss-free and the record stays thin.
* **Cross-pod replay.** When no live buffer exists on this pod (the producer is
  elsewhere, or the buffer was cleaned up), a follower polls that one snapshot
  and paces the freshly-grown content out as a smooth "fake completion",
  diffing by how much it has already metered out on this connection.
* **The lease fences a slow producer.** The supervision row is a per-conversation
  ownership lease (a monotonic ``epoch``). A pod that takes a presumed-dead turn
  over claims a higher epoch; the old producer — if it was only slow, not dead —
  self-fences the moment its guarded swaps start failing. Two pods never write
  one conversation.

The verdicts a follower acts on (``done`` / ``errored`` / ``dead``, plus the
out-of-band ``gone``) are defined on ``classify_supervision`` below.
"""

import asyncio
import math
import time
from dataclasses import dataclass
from typing import (
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    cast,
)

from zav.logging import logger
from zav.message_bus import HandlerException
from zav.pydantic_compat import PYDANTIC_V2, BaseModel

from zav.agents_sdk.adapters.stream_buffer.stream_buffer import (
    DEFAULT_CLEANUP_SECONDS,
    DEFAULT_INACTIVITY_SECONDS,
    DEFAULT_MAX_EVENTS_PER_BUFFER,
    RegisteredStream,
    StreamBuffer,
    StreamBufferRegistry,
    StreamSource,
)
from zav.agents_sdk.domain import ChatMessage
from zav.agents_sdk.domain.agent_async_worker import AgentAsyncWorker
from zav.agents_sdk.domain.chat_message import ContentPart
from zav.agents_sdk.domain.chat_stream_recording_store import (
    ChatStreamRecordingStore,
    RecordedStreamEvent,
)
from zav.agents_sdk.domain.chat_stream_supervision_store import (
    ChatStreamStatus,
    ChatStreamSupervision,
    ChatStreamSupervisionStore,
)
from zav.agents_sdk.handlers.commands import CreateChatStream

ModelT = TypeVar("ModelT", bound=BaseModel)

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 1.0
DEFAULT_DEAD_AFTER_SECONDS = 3.0
DEFAULT_FOLLOW_POLL_INTERVAL_SECONDS = 0.3
# How often the producer persists. Each streamed message is a full cumulative
# snapshot, so persisting coarsely just drops superseded intermediates; the
# live stream is untouched, only the durable recording is thinned.
DEFAULT_PERSIST_INTERVAL_SECONDS = 0.5
# Cap on the synthesized tick between two metered frames when pacing a coarse
# snapshot out as a live-looking stream. Smaller persist intervals mean each
# poll reveals a bigger jump; the follow loop spreads that jump across a few
# growing-prefix frames, sleeping up to this cap between them so the cushion
# outlasts one poll without freezing on a slow tool pause.
DEFAULT_REPLAY_MAX_GAP_SECONDS = 0.25
# How many growing-prefix frames one poll's new content is spread across when
# re-smoothing it into a live-looking stream. The whole batch is paced to drain
# over roughly one poll interval, so the next snapshot is fetched while this one
# is still streaming out — the consumer-side cushion.
DEFAULT_METER_FRAMES_PER_POLL = 12

# How many times a producer re-reads and re-swaps the supervision lease when a
# concurrent claim slips in between its read and its compare-and-swap. Lease
# claims only contend when two pods race the same conversation, so a few tries
# suffice.
CLAIM_MAX_ATTEMPTS = 3


class ChatStreamBusyError(HandlerException):
    """The conversation already has a fresh in-flight turn owned by another
    producer. Carries the occupying turn's id so the caller can attach to it
    instead of producing a second answer concurrently."""

    def __init__(self, session_id: str, running_message_id: Optional[str]):
        self.session_id = session_id
        self.running_message_id = running_message_id

    def __str__(self):
        return f"Conversation {self.session_id} already has a turn in flight" + (
            f" (message {self.running_message_id})" if self.running_message_id else ""
        )


class ChatStreamGoneError(HandlerException):
    """The conversation's stream slot no longer holds the requested turn — a
    newer turn overwrote it, or the sealed turn was reaped after its grace
    window. The caller must refetch the conversation, never re-drive the
    turn."""

    def __init__(self, session_id: str, message_id: str):
        self.session_id = session_id
        self.message_id = message_id

    def __str__(self):
        return (
            f"Stream for message {self.message_id} is gone from conversation "
            f"{self.session_id}; refetch the conversation"
        )


# The liveness verdict a consumer acts on for an in-flight turn: still
# ``running``, sealed ``done``, failed ``errored``, or its producer went
# ``dead`` (stale heartbeat). This is the consumer-facing type for slot
# discovery and status. ``gone`` (the slot moved on) is NOT part of it: it is
# surfaced as ``ChatStreamGoneError``, not returned as a status.
ChatStreamLiveness = Literal["running", "done", "errored", "dead"]

# What ``classify_supervision`` reports about a supervision row: a terminal
# verdict, or ``None`` while the turn is still being produced (i.e. still
# ``running``). It never returns ``running`` itself — absence of a verdict IS
# running — but it does return ``gone`` (the slot no longer holds this turn),
# which the caller turns into a ``ChatStreamGoneError`` rather than a status.
ClassifiedStreamVerdict = Literal["done", "errored", "dead", "gone"]


@dataclass
class ChatStreamSlotStatus:
    """What occupies a conversation's stream slot right now: the turn and a
    verdict (``running`` / ``done`` / ``errored`` / ``dead``). The discovery
    answer a client uses on load to re-attach to a still-running turn."""

    message_id: str
    status: ChatStreamLiveness


def classify_supervision(
    supervision: Optional[ChatStreamSupervision],
    message_id: str,
    now: float,
    dead_after_seconds: float,
) -> Optional[ClassifiedStreamVerdict]:
    """Map a conversation's supervision row to a verdict about ``message_id``,
    or ``None`` while that turn is still being produced.

    * ``gone``    — the slot no longer holds this turn: no row (reaped after
      sealing) or a newer turn overwrote it. Refetch the conversation — never
      re-drive, the turn already finished elsewhere.
    * ``done``    — sealed: replay to the end.
    * ``errored`` — the producer failed: replay what it managed, then stop.
    * ``dead``    — the owning pod stopped heart-beating: the turn is abandoned
      mid-flight and must be re-driven from the checkpoint, not replayed.
    """
    if supervision is None or supervision.message_id != message_id:
        return "gone"
    if supervision.status is ChatStreamStatus.DONE:
        return "done"
    if supervision.status is ChatStreamStatus.ERRORED:
        return "errored"
    if now - supervision.heartbeat_at > dead_after_seconds:
        return "dead"
    return None


class StreamMessageBus(Protocol):
    def handle_stream(
        self,
        message: CreateChatStream,
    ) -> AsyncGenerator[object, None]:
        raise NotImplementedError


class RecordingReplaySource:
    """Replays — and, when supervised, follows — a turn from the durable
    recording when no live buffer exists on this pod.

    This is the cross-pod / after-cleanup path (see the module docstring for
    the single-overwritten-snapshot / poll-and-pace mechanics): the producing
    pod's buffer is gone, but the shared recording still holds the turn.

    Because the slot belongs to the conversation, every poll checks the
    occupant: the moment the slot stops holding the followed turn (a newer
    turn overwrote it, or the sealed turn was reaped) the follow stops with
    ``gone`` — the consumer must refetch the conversation rather than keep
    pacing a snapshot that now belongs to a different turn.

    Without a supervision store it is a finite snapshot: pace the recorded
    answer to its end, then stop (``terminal_reason`` == "done"). With one it
    becomes a live follow: keep polling the snapshot while the producing pod
    heartbeats, and stop with a verdict the consumer can act on —

    * ``done``    — the producer sealed the turn; pace it to the end.
    * ``errored`` — the producer failed; pace what it managed, then stop.
    * ``dead``    — the producer stopped heart-beating (its pod died). The turn
      is abandoned mid-flight, so its recorded snapshot is a partial answer the
      consumer will re-drive from the agent's checkpoint. We stream none of it:
      we report ``dead`` and yield nothing further, leaving the consumer to
      take the turn over instead of replaying a corpse.
    * ``gone``    — the slot moved on: refetch the conversation, never
      re-drive.
    """

    def __init__(
        self,
        message_id: str,
        session_id: str,
        tenant: str,
        requester_uuid: Optional[str],
        recording_store: ChatStreamRecordingStore,
        supervision_store: Optional[ChatStreamSupervisionStore] = None,
        clock: Callable[[], float] = time.time,
        dead_after_seconds: float = DEFAULT_DEAD_AFTER_SECONDS,
        poll_interval_seconds: float = DEFAULT_FOLLOW_POLL_INTERVAL_SECONDS,
        replay_max_gap_seconds: float = DEFAULT_REPLAY_MAX_GAP_SECONDS,
        forced_terminal_reason: Optional[str] = None,
    ):
        self.__message_id = message_id
        self.__session_id = session_id
        self.__tenant = tenant
        self.__requester_uuid = requester_uuid
        self.__recording_store = recording_store
        self.__supervision_store = supervision_store
        self.__clock = clock
        self.__dead_after_seconds = dead_after_seconds
        self.__poll_interval_seconds = poll_interval_seconds
        self.__replay_max_gap_seconds = replay_max_gap_seconds
        self.__terminal_reason: Optional[str] = None
        self.__forced_terminal_reason: Optional[str] = forced_terminal_reason

    @property
    def message_id(self) -> str:
        return self.__message_id

    @property
    def terminal_reason(self) -> Optional[str]:
        if self.__forced_terminal_reason is not None:
            return self.__forced_terminal_reason
        return self.__terminal_reason

    async def iter_events(
        self, start_index: int = 0
    ) -> AsyncGenerator[Tuple[int, ChatMessage], None]:
        # The recording is one overwritten snapshot — the latest cumulative
        # message and the producer's yield index at that point. We poll it, and
        # each time the index advances we pace the new content out as a smooth
        # "fake completion", diffing by how much of the answer we have already
        # metered out on this connection. Metering one snapshot takes about a
        # poll interval, so the next snapshot is fetched while the current one
        # is still draining and the viewer never runs dry.
        emitted_index = start_index - 1
        emitted_chars = 0
        # A fresh viewer (start_index 0) has seen nothing, so its catch-up is
        # paced from the first character. A reconnecting viewer already showed
        # an earlier prefix on another pod; the single overwritten snapshot
        # cannot locate that prefix's offset, so its first read jumps straight
        # to the present (forward only, never a regression) and only the live
        # tail is paced from there.
        resuming = start_index > 0
        while True:
            # Classify BEFORE reading. A dead producer's snapshot is a corpse we
            # must not stream (the consumer re-drives from the checkpoint), and
            # a slot that moved on to a newer turn must surface as gone before
            # we pace the wrong turn's content — so the verdict has to gate the
            # read, not trail it. Reading after observing ``done`` still sees
            # the producer's final flushed snapshot (it persists before writing
            # the terminal status), so no separate tail-drain is needed.
            verdict = await self.__classify()
            if verdict in ("dead", "gone"):
                self.__terminal_reason = verdict
                return
            record = await self.__recording_store.read(
                tenant=self.__tenant,
                requester_uuid=self.__requester_uuid,
                session_id=self.__session_id,
            )
            if record is not None and record.message_id != self.__message_id:
                # Recording-only deployments have no supervision row to fence
                # with; the snapshot's own occupant stamp is the fallback
                # signal that the conversation moved on.
                self.__terminal_reason = "gone"
                return
            advanced = record is not None and record.index > emitted_index
            if record is not None and advanced:
                target = len(record.message.content)
                if resuming or target <= emitted_chars:
                    # Resume catch-up, or a structural-only advance (e.g. a tool
                    # part with no new answer text): hand over the whole
                    # snapshot at once rather than re-typing what is already on
                    # screen.
                    yield record.index, record.message
                    emitted_chars = target
                else:
                    async for frame_chars in self.__meter(emitted_chars, target):
                        yield record.index, self.__truncate(record.message, frame_chars)
                        emitted_chars = frame_chars
                emitted_index = record.index
                resuming = False
            if verdict is not None:
                self.__terminal_reason = verdict
                return
            if not advanced:
                await asyncio.sleep(self.__poll_interval_seconds)

    async def __meter(
        self, start_chars: int, target_chars: int
    ) -> AsyncGenerator[int, None]:
        # Spread one poll's new content across a few growing-prefix character
        # counts so a coarsely-persisted snapshot reads like live typing. The
        # batch is paced to drain over roughly one poll interval, capped per
        # frame so a large jump never freezes the viewer.
        span = target_chars - start_chars
        frames = min(span, DEFAULT_METER_FRAMES_PER_POLL)
        step = math.ceil(span / frames)
        tick = min(self.__poll_interval_seconds / frames, self.__replay_max_gap_seconds)
        emitted = start_chars
        while emitted < target_chars:
            emitted = min(target_chars, emitted + step)
            yield emitted
            if emitted < target_chars and tick > 0:
                await asyncio.sleep(tick)

    def __truncate(self, message: ChatMessage, n_chars: int) -> ChatMessage:
        # Rebuild the cumulative message as it looked at ``n_chars`` of answer
        # text: trim ``content`` and the trailing text part to the cut, keep the
        # atomic parts (tool / table / context) that precede it. Tool frames pop
        # in whole, exactly as they streamed live.
        if n_chars >= len(message.content) and message.content_parts is None:
            return message
        truncated_content = message.content[:n_chars]
        if message.content_parts is None:
            return self.__with_updates(message, {"content": truncated_content})
        new_parts: List[ContentPart] = []
        consumed = 0
        for part in message.content_parts:
            if part.type == "text" and part.text:
                remaining = n_chars - consumed
                if remaining <= 0:
                    break
                if len(part.text) <= remaining:
                    new_parts.append(part)
                    consumed += len(part.text)
                else:
                    new_parts.append(
                        self.__with_updates(part, {"text": part.text[:remaining]})
                    )
                    break
            else:
                new_parts.append(part)
        return self.__with_updates(
            message, {"content": truncated_content, "content_parts": new_parts}
        )

    @staticmethod
    def __with_updates(model: ModelT, updates: Dict[str, object]) -> ModelT:
        if PYDANTIC_V2:
            return model.model_copy(update=updates)
        return model.copy(update=updates)

    async def __classify(self) -> Optional[str]:
        # None → still being produced, keep following. Otherwise a verdict:
        # "done"/"errored" (replay to the end), "dead" (abandoned) or "gone"
        # (the conversation's slot no longer holds this turn).
        if self.__supervision_store is None:
            return "done"
        supervision = await self.__supervision_store.read(
            tenant=self.__tenant,
            requester_uuid=self.__requester_uuid,
            session_id=self.__session_id,
        )
        return classify_supervision(
            supervision, self.__message_id, self.__clock(), self.__dead_after_seconds
        )


class LiveStreamSource:
    """The producing side of a chat turn — the write twin of
    ``RecordingReplaySource``.

    It owns one producer task that drains the message bus into an in-memory
    ``StreamBuffer`` (the live fan-out for currently-connected readers), plus
    two ``AgentAsyncWorker``s that lift the durable writes off the producer's
    hot path:

    - the recording worker flushes the latest cumulative snapshot to the
      recording store at most once per interval, so a reconnecting pod can
      replay the turn;
    - the heartbeat worker stamps a fresh ``running`` supervision status on a
      timer, so a watching pod can tell "the producer is alive" from "the turn
      advanced" — two clocks, not one.

    Neither durable write blocks the producer: the producer only ever stages a
    snapshot in memory and appends to the buffer. Sealing (on any exit) flushes
    the recording, stops the heartbeat *without* a final tick, then writes the
    terminal status — in that order, so a late beat can never resurrect a
    finished turn. On cancellation the terminal status is left unwritten: the
    heartbeat goes stale and a reconnecting pod reads the turn as dead, which is
    what an abandoned turn is.

    The supervision lease (a monotonic ``epoch`` on the conversation's slot) is
    claimed by ``ResumableChatStreams.create`` BEFORE this source exists —
    refusing a busy conversation has to happen before any stream is handed to
    the client — and passed in; every supervision write here is guarded by it.
    That guard is what self-fences a slow producer when a newer owner takes the
    slot (see the module docstring's lease story).

    The conversation's slot rows are normally garbage-collected by the next
    turn's claim overwriting them. For the last turn of a conversation there
    is no next claim, so after a clean seal the buffer's cleanup timer (the
    same grace that forgets the in-memory buffer) also reaps the rows —
    guarded, so a slot already claimed by a newer turn is left alone.
    """

    def __init__(
        self,
        message_bus: StreamMessageBus,
        command: CreateChatStream,
        on_cleanup: Optional[Callable[[], None]] = None,
        recording_store: Optional[ChatStreamRecordingStore] = None,
        supervision_store: Optional[ChatStreamSupervisionStore] = None,
        epoch: int = 0,
        clock: Callable[[], float] = time.time,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        persist_interval_seconds: float = DEFAULT_PERSIST_INTERVAL_SECONDS,
        inactivity_seconds: float = DEFAULT_INACTIVITY_SECONDS,
        cleanup_seconds: float = DEFAULT_CLEANUP_SECONDS,
        max_events: int = DEFAULT_MAX_EVENTS_PER_BUFFER,
        max_run_seconds: Optional[float] = None,
    ):
        self.__message_bus = message_bus
        self.__command = command
        self.__tenant = command.tenant
        self.__requester_uuid = command.request_headers.requester_uuid
        self.__message_id = command.message_id
        self.__session_id = command.session_id
        self.__recording_store = recording_store
        self.__supervision_store = supervision_store
        self.__clock = clock
        self.__heartbeat_interval_seconds = heartbeat_interval_seconds
        self.__persist_interval_seconds = persist_interval_seconds
        # A turn runs to completion regardless of who is watching: the inactivity
        # timer no longer cancels the producer (see ``__ignore_inactivity``). A
        # runaway turn is bounded by ``max_run_seconds`` instead.
        self.__max_run_seconds = max_run_seconds
        self.__buffer = StreamBuffer(
            message_id=self.__message_id,
            tenant=self.__tenant,
            on_inactive=self.__ignore_inactivity,
            on_cleanup=self.__cleanup(on_cleanup),
            inactivity_seconds=inactivity_seconds,
            cleanup_seconds=cleanup_seconds,
            max_events=max_events,
        )
        # ``pending`` holds the latest cumulative snapshot the recording worker
        # has not yet flushed. Each message fully supersedes the previous one,
        # so keeping only the newest is correct and the recording stays thin.
        self.__pending: Optional[RecordedStreamEvent] = None
        self.__index = 0
        self.__recording_worker: Optional[AgentAsyncWorker] = None
        self.__heartbeat_worker: Optional[AgentAsyncWorker] = None
        # Our epoch in the conversation's supervision lease, claimed by the
        # caller before constructing us. Every supervision write is guarded by
        # it, so we self-fence (stop producing) the moment a newer owner takes
        # the slot over.
        self.__epoch = epoch
        # The terminal status we successfully sealed, if any — only a clean
        # DONE seal licenses the post-grace reap of the conversation's rows.
        self.__sealed_terminal: Optional[ChatStreamStatus] = None
        # The reap task, kept referenced so it is not garbage-collected
        # mid-await and so a failed reap surfaces via its done callback.
        self.__reap_task: Optional[asyncio.Task] = None
        self.__producer_task = asyncio.create_task(self.__run_producer())

    @property
    def message_id(self) -> str:
        return self.__message_id

    @property
    def terminal_reason(self) -> Optional[str]:
        return None

    def iter_events(
        self, start_index: int = 0
    ) -> AsyncGenerator[Tuple[int, ChatMessage], None]:
        return self.__buffer.iter_events(start_index=start_index)

    async def aclose(self) -> None:
        if self.__producer_task.done():
            return
        self.__producer_task.cancel()
        try:
            await self.__producer_task
        except asyncio.CancelledError:
            pass

    async def __ignore_inactivity(self) -> None:
        # A turn runs to completion regardless of who is watching. The buffer's
        # inactivity timer used to cancel the producer (the only thing that wired
        # reader presence to run lifetime); now it never does. The buffer is
        # still freed when the turn ends (mark_done -> cleanup); a runaway turn
        # is bounded by ``max_run_seconds``, not by reader presence.
        logger.info(
            f"Stream {self.__message_id} has no readers; producer continues to "
            "completion (reader presence does not bound run lifetime)"
        )

    async def __drain(self, gen: AsyncGenerator[ChatMessage, None]) -> None:
        async for message in gen:
            self.__stage(message)
            await self.__buffer.append(message)
            self.__index += 1
            if self.__buffer.event_count >= self.__buffer.max_events:
                logger.warning(
                    f"Stream {self.__message_id} reached max events "
                    f"({self.__buffer.max_events}), stopping producer"
                )
                return

    async def __run_producer(self) -> None:
        terminal: Optional[ChatStreamStatus] = None
        gen: Optional[AsyncGenerator[ChatMessage, None]] = None
        try:
            self.__start_workers()
            gen = cast(
                AsyncGenerator[ChatMessage, None],
                self.__message_bus.handle_stream(self.__command),
            )
            try:
                if self.__max_run_seconds is None:
                    await self.__drain(gen)
                else:
                    await asyncio.wait_for(
                        self.__drain(gen), timeout=self.__max_run_seconds
                    )
                terminal = ChatStreamStatus.DONE
            except asyncio.TimeoutError:
                # Run budget exceeded — a real terminal, not a silent done, so a
                # reader-less turn can't run forever.
                terminal = ChatStreamStatus.ERRORED
                logger.warning(
                    f"Stream {self.__message_id} exceeded its run budget "
                    f"({self.__max_run_seconds}s), stopping producer"
                )
                await self.__buffer.mark_error(
                    TimeoutError(f"run budget of {self.__max_run_seconds}s exceeded")
                )
            except asyncio.CancelledError:
                await self.__buffer.mark_cancelled()
                raise
            except Exception as e:
                terminal = ChatStreamStatus.ERRORED
                logger.error(
                    f"Error consuming stream {self.__message_id}: {e}",
                    exc_info=True,
                )
                await self.__buffer.mark_error(e)
        finally:
            if gen is not None:
                await gen.aclose()
            # Seal BEFORE marking the buffer done so the durable terminal write
            # lands while readers are still attached; ``mark_done`` then releases
            # them. On cancellation ``terminal`` is None — see class docstring.
            await self.__seal(terminal)
            await self.__buffer.mark_done()

    def __start_workers(self) -> None:
        if self.__recording_store is not None:
            self.__recording_worker = AgentAsyncWorker(
                self.__flush_recording,
                interval_seconds=self.__persist_interval_seconds,
            )
        if self.__supervision_store is not None:
            self.__heartbeat_worker = AgentAsyncWorker(
                self.__beat,
                interval_seconds=self.__heartbeat_interval_seconds,
            )

    def __stage(self, message: ChatMessage) -> None:
        if self.__recording_store is None:
            return
        self.__pending = RecordedStreamEvent(
            index=self.__index,
            message=message,
            message_id=self.__message_id,
            epoch=self.__epoch,
        )

    async def __flush_recording(self) -> None:
        event = self.__pending
        if self.__recording_store is None or event is None:
            return
        # Clear before awaiting: same event loop, so read+clear is atomic and a
        # snapshot staged during the put below is kept for the next tick. The
        # write carries our epoch; the store rejects it if a newer owner has
        # taken the slot over, so a fenced producer cannot overwrite the record.
        self.__pending = None
        await self.__recording_store.put(
            tenant=self.__tenant,
            requester_uuid=self.__requester_uuid,
            session_id=self.__session_id,
            event=event,
        )

    async def __beat(self) -> None:
        if not await self.__put_supervision(ChatStreamStatus.RUNNING):
            # Lost the lease: a newer producer took the turn over. Stop now so
            # two pods never write one turn. Cancel without awaiting — we are
            # inside the heartbeat task, and the producer's seal stops this
            # worker as it unwinds.
            self.__producer_task.cancel()

    async def __seal(self, terminal: Optional[ChatStreamStatus]) -> None:
        # Order matters: flush the recording (final cumulative snapshot), then
        # stop the heartbeat WITHOUT a final tick, then write the terminal
        # status. A heartbeat tick after the terminal write would stamp a fresh
        # ``running`` and make a finished turn look alive again. A terminal
        # write that loses the lease swap is dropped: the turn is no longer
        # ours to seal — and a seal that is not ours must not license a reap.
        if self.__recording_worker is not None:
            await self.__recording_worker.finish()
        if self.__heartbeat_worker is not None:
            await self.__heartbeat_worker.stop()
        if terminal is not None:
            if await self.__put_supervision(terminal):
                self.__sealed_terminal = terminal

    def __cleanup(self, on_cleanup: Optional[Callable[[], None]]) -> Callable[[], None]:
        # The buffer's cleanup timer is the grace window: viewers had
        # ``cleanup_seconds`` to catch the sealed turn's tail. After it,
        # forget the in-memory buffer (the caller's callback) and reap the
        # conversation's durable rows — the no-next-turn fallback for slots
        # the next claim would otherwise have overwritten.
        def run() -> None:
            if on_cleanup is not None:
                on_cleanup()
            self.__reap_task = asyncio.ensure_future(self.__reap())
            self.__reap_task.add_done_callback(self.__on_reap_done)

        return run

    def __on_reap_done(self, task: "asyncio.Task") -> None:
        self.__reap_task = None
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                f"Reaping stream {self.__message_id} failed: {task.exception()}",
                exc_info=task.exception(),
            )

    async def __reap(self) -> None:
        # Only a clean seal is reaped: an errored slot keeps its verdict for
        # reconnecting viewers, and an unsealed (cancelled/fenced) slot is not
        # ours to remove. Deletion is occupant-guarded, so a newer turn's
        # claim — or a takeover of this turn at a higher epoch — wins over a
        # late reap. Supervision goes first: a reader racing the reap then
        # sees "gone" (refetch) rather than a sealed row whose recording has
        # already vanished.
        if self.__sealed_terminal is not ChatStreamStatus.DONE:
            return
        if self.__supervision_store is not None:
            await self.__supervision_store.delete(
                tenant=self.__tenant,
                requester_uuid=self.__requester_uuid,
                session_id=self.__session_id,
                message_id=self.__message_id,
                epoch=self.__epoch,
            )
        if self.__recording_store is not None:
            await self.__recording_store.delete(
                tenant=self.__tenant,
                requester_uuid=self.__requester_uuid,
                session_id=self.__session_id,
                message_id=self.__message_id,
            )

    async def __put_supervision(self, status: ChatStreamStatus) -> bool:
        return await self.__swap_supervision(self.__epoch, status, self.__epoch)

    async def __swap_supervision(
        self,
        expected_epoch: Optional[int],
        status: ChatStreamStatus,
        epoch: int,
    ) -> bool:
        if self.__supervision_store is None:
            return True
        return await self.__supervision_store.compare_and_swap(
            tenant=self.__tenant,
            requester_uuid=self.__requester_uuid,
            session_id=self.__session_id,
            expected_epoch=expected_epoch,
            supervision=ChatStreamSupervision(
                status=status,
                heartbeat_at=self.__clock(),
                message_id=self.__message_id,
                epoch=epoch,
            ),
        )


class ResumableChatStreams:
    """Entry point for producing, re-attaching to, and inspecting chat
    streams, keyed two ways on purpose: live buffers by turn
    (``message_id``), durable slot rows by conversation (``session_id``).

    A conversation has at most one in-flight turn. ``create`` enforces that
    by claiming the conversation's supervision slot before any producer
    exists: a fresh RUNNING slot held by a different turn refuses the claim
    (``ChatStreamBusyError`` → 409), anything else — no row, a sealed row, a
    stale heartbeat, or the same turn resuming — is claimed by overwriting,
    which is also how the previous turn's rows get garbage-collected.
    """

    def __init__(
        self,
        stream_buffer_registry: StreamBufferRegistry,
        message_bus: StreamMessageBus,
        recording_store: Optional[ChatStreamRecordingStore] = None,
        supervision_store: Optional[ChatStreamSupervisionStore] = None,
        clock: Callable[[], float] = time.time,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        dead_after_seconds: float = DEFAULT_DEAD_AFTER_SECONDS,
        follow_poll_interval_seconds: float = DEFAULT_FOLLOW_POLL_INTERVAL_SECONDS,
        persist_interval_seconds: float = DEFAULT_PERSIST_INTERVAL_SECONDS,
        replay_max_gap_seconds: float = DEFAULT_REPLAY_MAX_GAP_SECONDS,
        max_run_seconds: Optional[float] = None,
    ):
        self.__stream_buffer_registry = stream_buffer_registry
        self.__message_bus = message_bus
        self.__recording_store = recording_store
        self.__supervision_store = supervision_store
        self.__clock = clock
        self.__heartbeat_interval_seconds = heartbeat_interval_seconds
        self.__dead_after_seconds = dead_after_seconds
        self.__follow_poll_interval_seconds = follow_poll_interval_seconds
        self.__persist_interval_seconds = persist_interval_seconds
        self.__replay_max_gap_seconds = replay_max_gap_seconds
        self.__max_run_seconds = max_run_seconds

    async def create(self, command: CreateChatStream) -> RegisteredStream:
        registry = self.__stream_buffer_registry
        registry.ensure_capacity(tenant=command.tenant)
        epoch = await self.__claim(command)
        source = LiveStreamSource(
            message_bus=self.__message_bus,
            command=command,
            on_cleanup=lambda: registry.remove(
                message_id=command.message_id,
                tenant=command.tenant,
                requester_uuid=command.request_headers.requester_uuid,
                expected_source=source,
            ),
            recording_store=self.__recording_store,
            supervision_store=self.__supervision_store,
            epoch=epoch,
            clock=self.__clock,
            heartbeat_interval_seconds=self.__heartbeat_interval_seconds,
            persist_interval_seconds=self.__persist_interval_seconds,
            max_run_seconds=self.__max_run_seconds,
        )
        registry.register(
            message_id=command.message_id,
            tenant=command.tenant,
            source=source,
            requester_uuid=command.request_headers.requester_uuid,
        )
        return source

    async def __claim(self, command: CreateChatStream) -> int:
        # Claim the conversation's slot BEFORE any producer exists, so a busy
        # conversation is refused with a status code instead of a stream that
        # errors after the fact. Decision table over the current row:
        #
        # * no row, or a sealed (done/errored) row, or a stale RUNNING row —
        #   claimable: overwrite with the next epoch. The overwrite IS the
        #   garbage collection of the previous turn's slot.
        # * fresh RUNNING row holding THIS turn — a resume escalation: bump
        #   the epoch, fencing the interrupted producer.
        # * fresh RUNNING row holding a DIFFERENT turn — the conversation is
        #   busy: refuse, carrying the occupant so the caller can attach.
        #
        # A concurrent claim can land between our read and our swap, so retry
        # a few times before giving up.
        if self.__supervision_store is None:
            return 0
        current: Optional[ChatStreamSupervision] = None
        for _ in range(CLAIM_MAX_ATTEMPTS):
            current = await self.__supervision_store.read(
                tenant=command.tenant,
                requester_uuid=command.request_headers.requester_uuid,
                session_id=command.session_id,
            )
            if (
                current is not None
                and current.message_id != command.message_id
                and classify_supervision(
                    current,
                    current.message_id,
                    self.__clock(),
                    self.__dead_after_seconds,
                )
                is None
            ):
                raise ChatStreamBusyError(
                    session_id=command.session_id,
                    running_message_id=current.message_id,
                )
            expected_epoch = current.epoch if current is not None else None
            next_epoch = (current.epoch + 1) if current is not None else 0
            if await self.__supervision_store.compare_and_swap(
                tenant=command.tenant,
                requester_uuid=command.request_headers.requester_uuid,
                session_id=command.session_id,
                expected_epoch=expected_epoch,
                supervision=ChatStreamSupervision(
                    status=ChatStreamStatus.RUNNING,
                    heartbeat_at=self.__clock(),
                    message_id=command.message_id,
                    epoch=next_epoch,
                ),
            ):
                return next_epoch
        raise ChatStreamBusyError(
            session_id=command.session_id,
            running_message_id=current.message_id if current is not None else None,
        )

    async def get(
        self,
        message_id: str,
        tenant: str,
        requester_uuid: Optional[str] = None,
        session_id: Optional[str] = None,
        partial: bool = False,
    ) -> Optional[StreamSource]:
        """Resolve a viewer onto a turn's stream.

        Same-pod live buffers are found by ``message_id`` alone. The cross-pod
        durable path needs the conversation key too: the slot is read by
        ``session_id`` and checked for occupancy. Returns ``None`` when there
        is nothing to stream and the turn should be re-driven (dead producer,
        or no durable path); raises ``ChatStreamGoneError`` when the slot
        moved on — the turn finished and its rows were overwritten or reaped,
        so the caller must refetch the conversation instead.
        """
        source = self.__stream_buffer_registry.get(
            message_id=message_id,
            tenant=tenant,
            requester_uuid=requester_uuid,
        )
        if source is not None:
            return source
        if self.__recording_store is None or session_id is None:
            return None
        if self.__supervision_store is not None:
            supervision = await self.__supervision_store.read(
                tenant=tenant,
                requester_uuid=requester_uuid,
                session_id=session_id,
            )
            verdict = classify_supervision(
                supervision, message_id, self.__clock(), self.__dead_after_seconds
            )
            if verdict == "gone":
                raise ChatStreamGoneError(session_id=session_id, message_id=message_id)
            if verdict == "dead":
                if not partial:
                    # Default path (live recovery): report absence so the
                    # caller re-drives from the checkpoint (404 → resume) rather
                    # than replaying a partial answer that can never finish.
                    return None
                # Partial-replay path: the caller wants to show the recorded
                # content to the user (e.g. after a cancel or a pod kill,
                # on reload). Drain the recording without following new events
                # (supervision_store=None) and surface "dead" as the terminal
                # verdict so the client knows this is an incomplete turn.
                return RecordingReplaySource(
                    message_id=message_id,
                    session_id=session_id,
                    tenant=tenant,
                    requester_uuid=requester_uuid,
                    recording_store=self.__recording_store,
                    supervision_store=None,
                    forced_terminal_reason="dead",
                    clock=self.__clock,
                    dead_after_seconds=self.__dead_after_seconds,
                    poll_interval_seconds=self.__follow_poll_interval_seconds,
                    replay_max_gap_seconds=self.__replay_max_gap_seconds,
                )
        return RecordingReplaySource(
            message_id=message_id,
            session_id=session_id,
            tenant=tenant,
            requester_uuid=requester_uuid,
            recording_store=self.__recording_store,
            supervision_store=self.__supervision_store,
            clock=self.__clock,
            dead_after_seconds=self.__dead_after_seconds,
            poll_interval_seconds=self.__follow_poll_interval_seconds,
            replay_max_gap_seconds=self.__replay_max_gap_seconds,
        )

    async def status(
        self,
        session_id: str,
        tenant: str,
        requester_uuid: Optional[str] = None,
    ) -> Optional[ChatStreamSlotStatus]:
        """Report which turn occupies the conversation's slot and its state —
        the discovery hook a client uses on load to find (and re-attach to) a
        turn still running from a previous page. ``None`` when there is no
        slot to report: nothing in flight and nothing recent to replay."""
        if self.__supervision_store is None:
            return None
        supervision = await self.__supervision_store.read(
            tenant=tenant,
            requester_uuid=requester_uuid,
            session_id=session_id,
        )
        if supervision is None:
            return None
        # Classifying the row against its OWN occupant can never yield ``gone``
        # (the slot trivially holds the message_id we pass) — so the verdict is
        # always a real liveness state here, with ``None`` meaning ``running``.
        verdict = classify_supervision(
            supervision,
            supervision.message_id,
            self.__clock(),
            self.__dead_after_seconds,
        )
        status: ChatStreamLiveness
        if verdict is None or verdict == "gone":
            status = "running"
        else:
            status = verdict
        return ChatStreamSlotStatus(
            message_id=supervision.message_id,
            status=status,
        )

    async def cancel(
        self,
        message_id: str,
        tenant: str,
        requester_uuid: Optional[str] = None,
    ) -> bool:
        return await self.__stream_buffer_registry.cancel_stream(
            message_id=message_id,
            tenant=tenant,
            requester_uuid=requester_uuid,
        )
