import gc

from zav.logging import logger


def cleanup_memory():
    gc.collect(2)
    try:
        import ctypes
        import ctypes.util
        import sys

        # Dynamically locate the C standard library (libc).
        # On Linux this is typically "libc.so.6", while on macOS it is "libc.dylib".
        libc_path = ctypes.util.find_library("c")
        if libc_path is None:
            raise OSError("Unable to locate the C standard library (libc)")

        libc = ctypes.CDLL(libc_path)

        # Try `malloc_trim` first (available on glibc/Linux).
        if hasattr(libc, "malloc_trim"):
            # MALLOC_TRIM(0) tries to release all free memory to the OS
            libc.malloc_trim(0)
        # Fallback for macOS: use `malloc_zone_pressure_relief` when present.
        elif sys.platform == "darwin" and hasattr(libc, "malloc_zone_pressure_relief"):
            # size_t malloc_zone_pressure_relief(malloc_zone_t zone, size_t goal);
            relief_fn = libc.malloc_zone_pressure_relief
            relief_fn.restype = ctypes.c_size_t
            relief_fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            relief_fn(None, 0)
        else:
            # If no known memory-release function is available, we silently continue.
            logger.debug(
                "No suitable libc memory-release function found on this platform"
            )
    except (OSError, AttributeError, ImportError) as e:
        logger.info(f"Error in cleanup_memory: {e}")
        pass
