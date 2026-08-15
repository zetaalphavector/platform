import asyncio
import json
import os
import select
import subprocess
import sys
import termios
import textwrap
import threading
import time
import tty
import uuid
from typing import (
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import httpx
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class PerfMetrics:
    def __init__(self) -> None:
        self.__start: Optional[float] = None
        self.__first: Optional[float] = None
        self.__last: Optional[float] = None
        self.__end: Optional[float] = None
        self.__gaps: List[float] = []
        self.__recover_at: Optional[float] = None
        self.__recover_first: Optional[float] = None
        self.count = 0

    def mark_start(self) -> None:
        if self.__start is None:
            self.__start = time.monotonic()

    def resume_timing(self) -> None:
        # A new stream is feeding us again, so unfreeze the elapsed clock.
        self.__end = None

    def mark_complete(self) -> None:
        # Freeze elapsed: the server closed the stream, so the turn is done
        # from the client's point of view and the clock should stop ticking.
        if self.__end is None:
            self.__end = time.monotonic()

    def mark_recover(self) -> None:
        self.__recover_at = time.monotonic()
        self.__recover_first = None
        # Don't let the kill+recover gap pollute the inter-message gap stats;
        # the recover latency is tracked separately as recover_gap.
        self.__last = None

    def record(self) -> None:
        now = time.monotonic()
        if self.__start is None:
            self.__start = now
        if self.__first is None:
            self.__first = now
        if self.__last is not None:
            self.__gaps.append(now - self.__last)
        if self.__recover_at is not None and self.__recover_first is None:
            self.__recover_first = now
        self.__last = now
        self.count += 1

    @property
    def ttft(self) -> Optional[float]:
        if self.__start is None or self.__first is None:
            return None
        return self.__first - self.__start

    @property
    def last_gap(self) -> Optional[float]:
        return self.__gaps[-1] if self.__gaps else None

    @property
    def elapsed(self) -> Optional[float]:
        if self.__start is None:
            return None
        end = self.__end if self.__end is not None else time.monotonic()
        return end - self.__start

    @property
    def recover_gap(self) -> Optional[float]:
        if self.__recover_at is None or self.__recover_first is None:
            return None
        return self.__recover_first - self.__recover_at

    def percentile(self, p: float) -> Optional[float]:
        if not self.__gaps:
            return None
        ordered = sorted(self.__gaps)
        k = int(round((p / 100.0) * (len(ordered) - 1)))
        k = max(0, min(len(ordered) - 1, k))
        return ordered[k]


class KeyboardReader:
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[str]"):
        self.__loop = loop
        self.__queue = queue
        self.__thread: Optional[threading.Thread] = None
        self.__stop = threading.Event()
        self.__old_term: Optional[list] = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        self.__old_term = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self.__thread = threading.Thread(target=self.__read_loop, daemon=True)
        self.__thread.start()

    def stop(self) -> None:
        self.__stop.set()
        if self.__old_term is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.__old_term)
            self.__old_term = None

    def __read_loop(self) -> None:
        fd = sys.stdin.fileno()
        while not self.__stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            # Read directly from the raw fd to bypass Python's buffered text
            # layer. With sys.stdin.read(1) the buffered reader pulls a whole
            # chunk into its private buffer, so select() stops reporting the fd
            # ready and queued keystrokes only surface on the next keypress —
            # which feels like lag / having to press and hold.
            data = os.read(fd, 64)
            if not data:
                continue
            self.__emit(data.decode(errors="ignore"))

    def __emit(self, chunk: str) -> None:
        i = 0
        while i < len(chunk):
            ch = chunk[i]
            if ch != "\x1b":
                self.__loop.call_soon_threadsafe(self.__queue.put_nowait, ch)
                i += 1
                continue
            # Assemble an escape sequence (arrow / page keys) into a single
            # token so the client treats Up as one event, not three bytes.
            seq = ch
            i += 1
            while i < len(chunk):
                nxt = chunk[i]
                seq += nxt
                i += 1
                if nxt.isalpha() or nxt == "~":
                    break
            if seq == "\x1b":
                continue
            self.__loop.call_soon_threadsafe(self.__queue.put_nowait, seq)


class PerfClient:
    def __init__(
        self,
        *,
        servers: Dict[str, str],
        tenant: str,
        agent_identifier: str,
        prompt: str,
        message_id: Optional[str] = None,
    ):
        self.__servers = dict(servers)
        self.__order = list(self.__servers.keys())
        self.__current = self.__order[0]
        self.__tenant = tenant
        self.__agent_identifier = agent_identifier
        self.__prompt = prompt
        # Ids are server-assigned and learned from the stream: every streamed
        # bot message carries its ``message_id`` (this turn) and, in stateful
        # mode, its ``session_id`` (the conversation). To attach to a turn
        # produced elsewhere both may be provided up front as a copied handle
        # (``session_id:message_id``); a bare message id attaches same-pod only.
        session_id, parsed_message_id = (
            self.__split_handle(message_id) if message_id is not None else (None, None)
        )
        self.__message_id: Optional[str] = parsed_message_id
        self.__session_id: Optional[str] = session_id
        self.__attach = message_id is not None
        self.__client = httpx.AsyncClient()
        self.__metrics = PerfMetrics()
        self.__content = ""
        self.__blocks: List[Tuple[str, str]] = []
        self.__scroll_top: Optional[int] = None
        self.__view_top = 0
        self.__view_max = 0
        self.__page = 10
        self.__content_before_recover: Optional[str] = None
        self.__branch: Optional[str] = None
        self.__status = "idle — press [s] to start"
        self.__done = False
        self.__terminal_reason: Optional[str] = None
        self.__error_body: Optional[str] = None
        self.__quit = False
        self.__last_index = -1
        self.__resumed_from: Optional[int] = None
        self.__consumer: Optional[asyncio.Task] = None
        self.__keys: "asyncio.Queue[str]" = asyncio.Queue()
        self.__keyboard: Optional[KeyboardReader] = None
        self.__console = Console()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self.__keyboard = KeyboardReader(loop, self.__keys)
        self.__keyboard.start()
        if self.__attach and self.__message_id:
            self.__status = (
                f"idle — press [r] to attach to msg {self.__message_id[:8]} "
                f"on {self.__current}"
            )
        else:
            self.__status = f"idle — press [s] to query {self.__current}"
        try:
            with Live(
                self.__render(),
                console=self.__console,
                refresh_per_second=12,
                screen=True,
            ) as live:
                while not self.__quit:
                    await self.__drain_keys()
                    live.update(self.__render())
                    await asyncio.sleep(0.05)
        finally:
            self.__keyboard.stop()
            await self.__cancel_consumer()
            await self.__client.aclose()

    def __current_url(self) -> str:
        return self.__servers[self.__current]

    def __body(self) -> Dict[str, object]:
        return {
            "agent_identifier": self.__agent_identifier,
            "conversation": [{"sender": "user", "content": self.__prompt}],
        }

    async def __switch_to(
        self, factory: Callable[[], Coroutine[object, object, None]]
    ) -> None:
        await self.__cancel_consumer()
        self.__consumer = asyncio.create_task(factory())

    async def __cancel_consumer(self) -> None:
        if self.__consumer is not None and not self.__consumer.done():
            self.__consumer.cancel()
            try:
                await self.__consumer
            except BaseException:
                pass

    async def __drain_keys(self) -> None:
        while not self.__keys.empty():
            ch = self.__keys.get_nowait()
            await self.__handle_key(ch)

    async def __handle_key(self, ch: str) -> None:
        if ch in ("q", "Q"):
            self.__quit = True
        elif ch in ("s", "S"):
            self.__status = f"starting turn on {self.__current}"
            await self.__switch_to(self.__do_start_turn)
        elif ch in ("k", "K"):
            asyncio.create_task(self.__kill_current())
        elif ch in ("c", "C"):
            await self.__drop_client()
        elif ch in ("r", "R", " "):
            self.__prepare_recover()
            await self.__switch_to(self.__do_recover)
        elif ch in ("y", "Y"):
            self.__copy_message_id()
        elif ch in ("p", "P"):
            self.__paste_message_id()
        elif ch == "\x1b[A":
            self.__scroll_by(-1)
        elif ch == "\x1b[B":
            self.__scroll_by(1)
        elif ch == "\x1b[5~":
            self.__scroll_by(-self.__page)
        elif ch == "\x1b[6~":
            self.__scroll_by(self.__page)
        elif ch.isdigit():
            self.__aim(int(ch))

    def __scroll_by(self, delta: int) -> None:
        # Move an absolute top-line anchor. A positive delta scrolls down. Going
        # to (or past) the bottom re-engages tail-follow (scroll_top=None) so
        # new tokens keep streaming into view; otherwise the anchor stays fixed
        # on the same line even as more text streams in below it.
        base = self.__view_top if self.__scroll_top is None else self.__scroll_top
        target = base + delta
        if target >= self.__view_max:
            self.__scroll_top = None
        else:
            self.__scroll_top = max(0, target)

    def __copy_message_id(self) -> None:
        message_id = self.__message_id
        if message_id is None:
            self.__status = "no message id yet (start a turn first)"
            return
        # Copy the whole conversation handle, not just the turn id. A cross-pod
        # attach resolves the durable recording (and checkpoint) by
        # ``session_id``; with only the message id the other pod can't find the
        # conversation and re-runs the turn from scratch. Encode both as
        # ``session_id:message_id`` — a lone id (pre-stateful) stays a bare
        # message id.
        handle = (
            f"{self.__session_id}:{message_id}"
            if self.__session_id is not None
            else message_id
        )
        try:
            subprocess.run(["pbcopy"], input=handle.encode(), check=True)
            self.__status = (
                f"copied handle (sess {self.__session_id[:8]} / msg "
                f"{message_id[:8]}) to clipboard"
                if self.__session_id is not None
                else f"copied msg {message_id[:8]} (no session yet) to clipboard"
            )
        except (OSError, subprocess.SubprocessError):
            self.__status = f"handle: {handle} (copy failed)"

    @staticmethod
    def __split_handle(text: str) -> Tuple[Optional[str], str]:
        # A copied handle is ``session_id:message_id``; a lone uuid is a message
        # id with no conversation key (resolves only on the producing pod).
        # uuids never contain a colon, so split on the last one.
        session_part, _, message_part = text.strip().rpartition(":")
        message_id = str(uuid.UUID(message_part))
        session_id = str(uuid.UUID(session_part)) if session_part else None
        return session_id, message_id

    def __paste_message_id(self) -> None:
        try:
            out = subprocess.run(["pbpaste"], capture_output=True, check=True)
        except (OSError, subprocess.SubprocessError):
            self.__status = "paste failed (pbpaste unavailable)"
            return
        text = out.stdout.decode(errors="ignore").strip()
        try:
            session_id, message_id = self.__split_handle(text)
        except ValueError:
            self.__status = f"clipboard is not a valid handle: {text[:48]!r}"
            return
        self.__message_id = message_id
        self.__session_id = session_id
        self.__attach = True
        if session_id is not None:
            self.__status = (
                f"pasted handle (sess {session_id[:8]} / msg {message_id[:8]}) "
                f"— press [r] to attach on {self.__current}"
            )
        else:
            self.__status = (
                f"pasted msg {message_id[:8]} (no session — same-pod only) "
                f"— press [r] to attach on {self.__current}"
            )

    def __aim(self, n: int) -> None:
        if 1 <= n <= len(self.__order):
            self.__current = self.__order[n - 1]
            self.__status = f"aimed at {self.__current}"

    async def __drop_client(self) -> None:
        await self.__cancel_consumer()
        # The client stops watching, so its elapsed clock should stop too; the
        # server keeps producing in the background and we can re-attach later
        # with recover, which unfreezes the clock again.
        self.__metrics.mark_complete()
        self.__status = f"client view dropped ({self.__current} still running)"

    def __prepare_recover(self) -> None:
        self.__content_before_recover = self.__content
        self.__branch = None
        self.__resumed_from = self.__last_index + 1
        self.__scroll_top = None
        self.__metrics.mark_recover()
        self.__status = f"recovering on {self.__current}…"

    async def __kill_current(self) -> None:
        label = self.__current
        url = self.__servers[label] + "/perf/kill"
        self.__status = f"killing {label}…"
        try:
            await self.__client.post(url, timeout=2.0)
        except httpx.HTTPError:
            pass
        self.__status = f"killed {label} (server process exited)"

    def __reset_for_new_turn(self) -> None:
        # [s] starts a brand-new turn: the server assigns the ids (learned from
        # the first streamed message), fresh metrics, and no leftover content
        # or resume markers from the previous turn.
        self.__message_id = None
        self.__session_id = None
        self.__metrics = PerfMetrics()
        self.__content = ""
        self.__blocks = []
        self.__scroll_top = None
        self.__content_before_recover = None
        self.__branch = None
        self.__resumed_from = None
        self.__last_index = -1
        self.__done = False

    async def __do_start_turn(self) -> None:
        self.__reset_for_new_turn()
        await self.__consume(
            "POST",
            "/chat/stream",
            params={"tenant": self.__tenant},
            json_body={**self.__body(), "stateful": True},
            branch="initial",
            reset_index=True,
        )

    async def __do_recover(self) -> None:
        # One recovery flow: the client only knows its stream dropped, not why.
        # First try to re-attach to a live producer on the current server. A
        # 404 then means one of two things, told apart by whether the turn had
        # already finished. If it was still in flight the producer is gone
        # (killed, or living on another server), so resume from the last
        # checkpoint. If it had already completed the buffer was merely cleaned
        # up after its TTL; there is nothing left to stream, so don't start a
        # fresh turn — that would just re-run the agent.
        was_done = self.__done
        message_id = self.__message_id
        if message_id is None:
            # No event ever arrived, so there is no turn to address: nothing
            # was rendered and nothing can be re-attached or resumed. Start
            # over with [s] instead.
            self.__status = "nothing to recover (no turn id learned yet)"
            return
        code = await self.__consume(
            "GET",
            f"/chat/stream/{message_id}",
            params={
                "tenant": self.__tenant,
                "start_index": self.__last_index + 1,
                **({"session_id": self.__session_id} if self.__session_id else {}),
            },
            json_body=None,
            branch="attached (GET 200)",
            reset_index=False,
        )
        if code == 410:
            # The conversation's slot moved past this turn: it finished and
            # its rows were overwritten or reaped. Refetch the conversation
            # (here: nothing to stream) — re-driving would re-run a turn that
            # already completed.
            self.__metrics.mark_complete()
            self.__done = True
            self.__resumed_from = None
            self.__branch = "gone (410)"
            self.__status = "turn already completed; refetch the conversation (410)"
            return
        if code == 404:
            if was_done:
                self.__metrics.mark_complete()
                self.__done = True
                self.__resumed_from = None
                self.__branch = "complete (nothing to recover)"
                self.__status = "already complete; nothing to recover (404)"
                return
            self.__status = f"no live stream on {self.__current}; resuming…"
            # If the session id was never observed (attach to a foreign turn
            # that 404s before any event arrives) resume without one: the
            # server mints a fresh session and, with no checkpoint under it,
            # re-runs the turn from the incoming message under the same
            # message id.
            code = await self.__consume(
                "POST",
                "/chat/stream",
                params={
                    "tenant": self.__tenant,
                    "resume": "true",
                    "message_id": message_id,
                },
                json_body={
                    **self.__body(),
                    "stateful": True,
                    **({"session_id": self.__session_id} if self.__session_id else {}),
                },
                branch="resumed (POST after 404)",
                reset_index=True,
            )
            if code == 409:
                # Another turn of this conversation is in flight — attach to
                # the occupant the server told us about instead of racing it.
                occupant = self.__parse_busy_occupant()
                if occupant is None:
                    self.__status = "conversation busy (409); no occupant id"
                    return
                self.__message_id = occupant
                await self.__consume(
                    "GET",
                    f"/chat/stream/{occupant}",
                    params={
                        "tenant": self.__tenant,
                        "start_index": 0,
                        **(
                            {"session_id": self.__session_id}
                            if self.__session_id
                            else {}
                        ),
                    },
                    json_body=None,
                    branch="attached (busy occupant)",
                    reset_index=True,
                )

    def __parse_busy_occupant(self) -> Optional[str]:
        if self.__error_body is None:
            return None
        try:
            detail = json.loads(self.__error_body).get("detail")
            occupant = detail.get("message_id") if isinstance(detail, dict) else None
        except (json.JSONDecodeError, AttributeError):
            return None
        return occupant if isinstance(occupant, str) else None

    async def __consume(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Union[str, int]],
        json_body: Optional[Dict[str, object]],
        branch: str,
        reset_index: bool,
    ) -> Optional[int]:
        url = self.__current_url() + path
        self.__metrics.mark_start()
        self.__metrics.resume_timing()
        self.__done = False
        if reset_index:
            self.__last_index = -1
        try:
            async with self.__client.stream(
                method, url, params=params, json=json_body, timeout=None
            ) as resp:
                if resp.status_code == 404:
                    return 404
                if resp.status_code >= 400:
                    body = await resp.aread()
                    self.__error_body = body.decode("utf-8", errors="replace")
                    self.__status = f"{method} {path} -> {resp.status_code}"
                    return resp.status_code
                self.__branch = branch
                self.__status = f"streaming from {self.__current} [{branch}]"
                self.__terminal_reason = None
                async for event_id, event_type, payload in self.__iter_sse(resp):
                    if event_type == "stream_status":
                        self.__terminal_reason = self.__parse_terminal_reason(payload)
                        continue
                    self.__apply_event(event_id, payload)
                return self.__settle_stream_close(resp.status_code)
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as e:
            self.__status = f"stream lost on {self.__current}: {type(e).__name__}"
            return None

    def __settle_stream_close(self, status_code: int) -> int:
        # How a closed stream reads depends on the terminal control event, if
        # any. A live buffer sends none, so a clean close simply means the turn
        # finished. A followed recording sends a verdict so we can tell a sealed
        # turn from one whose producing pod died (abandoned, still resumable) or
        # one whose producer failed.
        if self.__terminal_reason == "dead":
            self.__metrics.mark_complete()
            self.__done = False
            self.__status = (
                f"producer died — turn abandoned on {self.__current}, "
                f"press [r] to resume"
            )
            return status_code
        if self.__terminal_reason == "errored":
            self.__metrics.mark_complete()
            self.__done = True
            self.__status = f"errored on {self.__current} (producer failed)"
            return status_code
        if self.__terminal_reason == "gone":
            self.__metrics.mark_complete()
            self.__done = True
            self.__status = (
                "turn gone mid-follow (conversation moved on); "
                "refetch the conversation"
            )
            return status_code
        self.__metrics.mark_complete()
        self.__done = True
        self.__status = f"done on {self.__current} (stream closed)"
        return status_code

    def __parse_terminal_reason(self, payload: str) -> Optional[str]:
        try:
            return json.loads(payload).get("reason")
        except (json.JSONDecodeError, AttributeError):
            return None

    async def __iter_sse(
        self, resp: httpx.Response
    ) -> AsyncIterator[Tuple[Optional[str], Optional[str], str]]:
        data_lines: List[str] = []
        event_id: Optional[str] = None
        event_type: Optional[str] = None
        async for raw in resp.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if data_lines:
                    yield event_id, event_type, "\n".join(data_lines)
                data_lines = []
                event_id = None
                event_type = None
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
            elif line.startswith("id:"):
                event_id = line[len("id:") :].strip()
            elif line.startswith("event:"):
                event_type = line[len("event:") :].strip()

    def __apply_event(self, event_id: Optional[str], payload: str) -> None:
        if event_id and ":" in event_id:
            try:
                self.__last_index = int(event_id.rsplit(":", 1)[1])
            except ValueError:
                pass
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            return
        message_id = message.get("message_id")
        if isinstance(message_id, str) and message_id:
            self.__message_id = message_id
        session_id = message.get("session_id")
        if isinstance(session_id, str) and session_id:
            self.__session_id = session_id
        content = message.get("content")
        if isinstance(content, str):
            self.__content = content
        self.__blocks = self.__build_blocks(message)
        self.__metrics.record()

    def __build_blocks(self, message: Dict) -> List[Tuple[str, str]]:
        # Turn one streamed ChatMessage into ordered (text, style) blocks: tool
        # calls interleaved with the answer text, mirroring how ui_app's
        # render_tool_content surfaces each tool part.
        blocks: List[Tuple[str, str]] = []
        has_text = False
        parts = message.get("content_parts")
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                tool = part.get("tool")
                if part.get("type") == "tool" and isinstance(tool, dict):
                    blocks.append((self.__tool_text(tool), self.__tool_style(tool)))
                elif part.get("type") == "text" and part.get("text"):
                    blocks.append((part["text"], "default"))
                    has_text = True
        content = message.get("content")
        if isinstance(content, str) and content and not has_text:
            blocks.append((content, "default"))
        return blocks

    @staticmethod
    def __tool_text(tool: Dict) -> str:
        name = tool.get("name") or "tool"
        status = str(tool.get("status") or "")
        mark = {"running": "⟳", "completed": "✓", "error": "✗"}.get(status, "•")
        display = tool.get("display_text")
        if display:
            return f"{mark} {display}"
        text = f"{mark} {name}"
        params = tool.get("params") or {}
        hint = params.get("query") or params.get("message")
        if isinstance(hint, str) and hint:
            text += f" — {hint}"
        return text

    @staticmethod
    def __tool_style(tool: Dict) -> str:
        return {
            "running": "yellow",
            "completed": "green",
            "error": "bold red",
        }.get(str(tool.get("status") or ""), "magenta")

    def __render(self) -> RenderableType:
        header = Text()
        header.append(f"server={self.__current}  ", style="bold cyan")
        header.append(f"msg={(self.__message_id or '—')[:8]}  ", style="dim")
        header.append(f"sess={(self.__session_id or '—')[:8]}  ", style="dim")
        header.append(self.__status, style="bold green" if self.__done else "yellow")

        footer = self.__build_footer()
        width, height = self.__console.size
        footer_height = self.__measure_height(footer, width)
        content_region = max(3, height - footer_height)
        # Crop the streamed answer to the panel's visible tail so a long answer
        # scrolls inside the box instead of pushing the metrics footer off the
        # screen. The 4/6 account for the panel border plus padding=(1, 2).
        inner_height = max(1, content_region - 4)
        inner_width = max(10, width - 6)
        self.__page = max(1, inner_height - 1)
        if self.__blocks:
            body, scroll_hint = self.__build_body(inner_width, inner_height)
        else:
            placeholder = (
                "(press [s] to start the turn)"
                if self.__metrics.elapsed is None
                else "(waiting for first message…)"
            )
            body, scroll_hint = Text(placeholder, style="dim"), ""
        if scroll_hint:
            header.append(f"  {scroll_hint}", style="dim")
        content_panel = Panel(body, title=header, border_style="cyan", padding=(1, 2))

        layout = Layout()
        layout.split_column(
            Layout(content_panel, name="content"),
            Layout(footer, name="footer", size=footer_height),
        )
        return layout

    def __build_footer(self) -> RenderableType:
        metrics = self.__metrics
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold")
        table.add_column()
        table.add_row("time to first msg", self.__fmt_ms(metrics.ttft))
        table.add_row("last gap", self.__fmt_ms(metrics.last_gap))
        table.add_row(
            "p50 / p95 gap",
            f"{self.__fmt_ms(metrics.percentile(50))}"
            f" / {self.__fmt_ms(metrics.percentile(95))}",
        )
        table.add_row("events", str(metrics.count))
        table.add_row("chars", str(len(self.__content)))
        table.add_row("stream idx", str(self.__last_index))
        table.add_row("elapsed", self.__fmt_ms(metrics.elapsed))
        if self.__branch is not None:
            table.add_row("stream source", self.__branch)
        if self.__resumed_from is not None:
            table.add_row("resumed from idx", str(self.__resumed_from))
        if metrics.recover_gap is not None:
            table.add_row("recover gap", self.__fmt_ms(metrics.recover_gap))
        if self.__content_before_recover is not None:
            table.add_row(
                "chars @recover",
                f"{len(self.__content_before_recover)} -> {len(self.__content)}",
            )

        servers_line = Text("servers: ")
        for i, label in enumerate(self.__order, start=1):
            style = "bold green" if label == self.__current else "dim"
            servers_line.append(f"[{i}] {label}  ", style=style)

        help_line = Text(
            "keys: [s]tart  [1..] aim  [k]ill  [c]lient-drop  "
            "[r/space] recover  [y]ank-id  [p]aste-id  "
            "[↑↓/PgUp/PgDn] scroll  [q]uit",
            style="dim",
        )

        return Group(
            Padding(table, (1, 0, 0, 2)),
            Padding(servers_line, (1, 0, 0, 2)),
            Padding(help_line, (0, 0, 0, 2)),
        )

    def __measure_height(self, renderable: RenderableType, width: int) -> int:
        options = self.__console.options.update(width=width, height=None)
        return len(self.__console.render_lines(renderable, options, pad=False))

    def __build_body(self, width: int, height: int) -> Tuple[Text, str]:
        # Wrap every block to the panel width, then show a height-sized window
        # into the wrapped lines. __scroll_top is an absolute top-line anchor:
        # None follows the live tail, an int pins that line to the top so the
        # view stays put while more text streams in below.
        phys: List[Tuple[str, str]] = []
        for i, (text, style) in enumerate(self.__blocks):
            if i > 0:
                phys.append(("", "default"))
            for paragraph in text.split("\n"):
                if not paragraph:
                    phys.append(("", style))
                    continue
                wrapped = textwrap.wrap(
                    paragraph,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                )
                for line in wrapped or [""]:
                    phys.append((line, style))
        total = len(phys)
        max_top = max(0, total - height)
        if self.__scroll_top is None:
            start = max_top
        else:
            start = min(self.__scroll_top, max_top)
        end = min(total, start + height)
        self.__view_top = start
        self.__view_max = max_top
        body = Text()
        for j, (line, style) in enumerate(phys[start:end]):
            body.append(line, style=None if style == "default" else style)
            if start + j < end - 1:
                body.append("\n")
        hint = ""
        if self.__scroll_top is not None and start < max_top:
            hint = f"↑{start} above · ↓{total - end} below"
        return body, hint

    @staticmethod
    def __fmt_ms(value: Optional[float]) -> str:
        if value is None:
            return "—"
        ms = value * 1000
        if ms < 1:
            return f"{value * 1_000_000:.0f} µs"
        if ms < 10:
            return f"{ms:.1f} ms"
        if ms < 1000:
            return f"{ms:.0f} ms"
        return f"{value:.2f} s"


def run_client(
    *,
    servers: Dict[str, str],
    tenant: str,
    agent_identifier: str,
    prompt: str,
    message_id: Optional[str] = None,
) -> None:
    client = PerfClient(
        servers=servers,
        tenant=tenant,
        agent_identifier=agent_identifier,
        prompt=prompt,
        message_id=message_id,
    )
    asyncio.run(client.run())
