import asyncio
import functools
import inspect
from concurrent.futures import ThreadPoolExecutor


def is_bound_function(obj):
    return inspect.ismethod(obj) and callable(obj)


def force_async(fn):
    """Turns a sync function to async function using threads."""
    pool = ThreadPoolExecutor()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)  # make it awaitable

    return wrapper
