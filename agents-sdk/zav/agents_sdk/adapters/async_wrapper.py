import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, TypeVar

from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def asyncify(f: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """Turns a sync function to async function using threads."""
    pool = ThreadPoolExecutor()

    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        future = pool.submit(f, *args, **kwargs)
        return asyncio.wrap_future(future)  # make it awaitable

    return wrapper
