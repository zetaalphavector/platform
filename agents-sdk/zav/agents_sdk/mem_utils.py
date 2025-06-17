import gc

from zav.logging import logger


def cleanup_memory():
    gc.collect(2)
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        # MALLOC_TRIM(0) tries to release all free memory to the OS
        libc.malloc_trim(0)
    except (OSError, AttributeError, ImportError) as e:  # pragma: no cover
        logger.info(f"Error in cleanup_memory: {e}")
        pass
