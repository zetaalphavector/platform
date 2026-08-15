import atexit
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, Optional

from zav.logging import logger
from zav.pydantic_compat import PYDANTIC_V2

from zav.llm_tracing.trace import Span


@lru_cache(maxsize=1)
def __executor() -> ThreadPoolExecutor:
    # A single worker keeps tracing-backend calls off the caller's event loop
    # while preserving submission order, so the backend's observation
    # create/update ordering and internal state stay consistent without locks.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-tracing")
    # Drain queued calls at interpreter exit so spans submitted right before
    # shutdown still reach the backend. Registered lazily on first use — i.e.
    # after the tracing backend's own client has been constructed — so atexit's
    # LIFO order runs this drain *before* that client's flush (e.g. Langfuse's
    # TaskManager registers its flush at construction). Draining first lets the
    # offloaded calls enqueue into the backend before it flushes; otherwise the
    # last spans would be lost.
    atexit.register(__drain_at_exit)
    return executor


def __drain_at_exit() -> None:
    try:
        flush_tracing_calls(timeout=5)
    except Exception:
        logger.warning("Timed out flushing offloaded tracing calls during shutdown")


def __snapshot(span: Span) -> Span:
    # Shallow-copy the mutable containers the backend reads, so the worker
    # serializes a frozen view even if the caller mutates the live span next
    # (e.g. a follow-up span.update() while this one is being serialized).
    update = {"attributes": dict(span.attributes), "events": list(span.events)}
    return span.model_copy(update=update) if PYDANTIC_V2 else span.copy(update=update)


def __run_safely(fn: Callable[[Span], None], span: Span) -> None:
    try:
        fn(span)
    except Exception:
        logger.exception(
            "Tracing backend call failed for span %s", span.context.span_id
        )


def submit_tracing_call(fn: Callable[[Span], None], span: Span) -> None:
    """Run a tracing-backend call on the shared worker thread instead of the
    caller's thread. The span is snapshotted on the calling thread, so the slow
    work (payload serialization) never blocks the caller's event loop."""
    __executor().submit(__run_safely, fn, __snapshot(span))


def flush_tracing_calls(timeout: Optional[float] = None) -> None:
    """Block until all previously submitted tracing calls have finished. Useful
    for graceful shutdown and for deterministic tests."""
    __executor().submit(lambda: None).result(timeout)
