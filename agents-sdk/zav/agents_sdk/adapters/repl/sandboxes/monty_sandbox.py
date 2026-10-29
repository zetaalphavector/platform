import os
from typing import Any, Dict, List

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.repl.repl_sandbox import HostFunctions, ReplSandbox
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

WORKSPACE_VIRTUAL_PATH = "/workspace"


class MontyReplSandboxConfiguration(BaseModel):
    enabled: bool = Field(
        False, description="Run `run_python` in the in-process Monty sandbox."
    )
    workspace_directory: str = Field(
        ".workspace",
        description=(
            "Host directory mounted read-write at /workspace. Must contain only "
            "data: anything written there is untrusted and must never be executed "
            "by the host."
        ),
    )
    read_only_mounts: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra mounts as {virtual_path: host_directory}, read-only. Use it for "
            "docs and the attempts directory."
        ),
    )
    max_duration_seconds: float = Field(
        60.0,
        description=(
            "Cumulative CPU time budget for one REPL session. When exhausted the "
            "session is recreated and its variables are lost."
        ),
    )
    max_memory_bytes: int = Field(256_000_000, description="Sandbox memory cap.")


# pydantic.dev/docs/monty/limitations, so the model does not rediscover the
# missing modules one failed import at a time.
_PYTHON_SUBSET = (
    "The interpreter runs a Python subset. Available standard library: asyncio "
    "(no event loop; `await` at top level, `asyncio.gather` for parallel tool "
    "calls), base64, binascii, collections, dataclasses, datetime, functools, "
    "itertools, json, math, os, pathlib, re, sys, typing, unicodedata. Absent: "
    "textwrap, enum, contextlib, random, time, io, copy, string, struct, "
    "operator, inspect, logging, traceback, hashlib, uuid, urllib and anything "
    "network or process related; no third-party packages. No class inheritance, "
    "no @classmethod/@staticmethod/@property, no generators (`yield`), no "
    "`match`, no `eval`/`exec`; enumerate, zip, map and filter are eager. Do "
    "string work with str methods and slicing; do not import what you do not "
    "need."
)


class MontyReplSandbox(ReplSandbox):
    """pydantic-monty: a Rust Python interpreter with no network, no environment
    and no host filesystem beyond the mounts. Installed with agents-sdk[repl]."""

    source_name = "monty"
    source_description = (
        "A sandboxed Python interpreter with no network or file access."
    )

    def __init__(
        self,
        enabled: bool,
        workspace_directory: str,
        read_only_mounts: Dict[str, str],
        max_duration_seconds: float,
        max_memory_bytes: int,
    ) -> None:
        self.enabled = enabled
        self.__workspace_directory = workspace_directory
        self.__read_only_mounts = read_only_mounts
        self.__max_duration_seconds = max_duration_seconds
        self.__max_memory_bytes = max_memory_bytes
        self.__pool: Any = None
        self.__session_context: Any = None
        self.__session: Any = None

    async def to_prompt(self) -> str:
        mounts = ", ".join(f"`{path}`" for path in self.__read_only_mounts)
        read_only = f" {mounts} are mounted read-only." if mounts else ""
        return (
            "The REPL is sandboxed: no network, no environment variables and no "
            f"host filesystem beyond the mounts. `{WORKSPACE_VIRTUAL_PATH}` is "
            f"writable and persists between calls.{read_only} The session has a "
            f"cumulative CPU budget of {self.__max_duration_seconds:g}s; when it is "
            "spent the session is recreated and all variables are lost.\n\n"
            + _PYTHON_SUBSET
        )

    async def run(self, code: str, host_functions: HostFunctions) -> str:
        try:
            import pydantic_monty
        except ImportError:
            return (
                "Error: the Python tool needs the `pydantic-monty` package "
                "(install agents-sdk[repl])."
            )
        session = await self.__ensure_session(pydantic_monty)
        printed = pydantic_monty.CollectStreams()
        try:
            value = await session.feed_run(
                code,
                external_lookup=host_functions,
                mount=self.__mounts(pydantic_monty),
                print_callback=printed,
            )
        except pydantic_monty.MontySyntaxError as e:
            return f"SyntaxError: {e}"
        except pydantic_monty.MontyRuntimeError as e:
            message = str(e)
            if "time limit exceeded" in message:
                await self.__reset_session()
                message += (
                    "\nThe REPL's time budget is exhausted; the session was reset "
                    "and all variables are gone."
                )
            return f"Error: {message}"
        except Exception as e:
            logger.error(f"run_python failed: {e}")
            await self.__reset_session()
            return f"Error: {type(e).__name__}: {e}"
        return self.__render_output(printed.output, value)

    async def aclose(self) -> None:
        await self.__reset_session()
        if self.__pool is not None:
            pool, self.__pool = self.__pool, None
            try:
                await pool.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing the Python sandbox pool: {e}")

    def __mounts(self, pydantic_monty: Any) -> List[Any]:
        os.makedirs(self.__workspace_directory, exist_ok=True)
        mounts = [
            pydantic_monty.MountDir(
                host_path=self.__workspace_directory,
                virtual_path=WORKSPACE_VIRTUAL_PATH,
                mode="read-write",
            )
        ]
        for virtual_path, host_path in self.__read_only_mounts.items():
            if not os.path.isdir(host_path):
                continue
            mounts.append(
                pydantic_monty.MountDir(
                    host_path=host_path, virtual_path=virtual_path, mode="read-only"
                )
            )
        return mounts

    async def __ensure_session(self, pydantic_monty: Any) -> Any:
        if self.__session is not None:
            return self.__session
        if self.__pool is None:
            self.__pool = await pydantic_monty.AsyncMonty().__aenter__()
        self.__session_context = self.__pool.checkout(
            limits={
                "max_duration_secs": self.__max_duration_seconds,
                "max_memory": self.__max_memory_bytes,
            }
        )
        self.__session = await self.__session_context.__aenter__()
        return self.__session

    async def __reset_session(self) -> None:
        if self.__session_context is None:
            return
        context, self.__session_context = self.__session_context, None
        self.__session = None
        try:
            await context.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error closing the Python sandbox session: {e}")

    @staticmethod
    def __render_output(printed: List[Any], value: Any) -> str:
        parts: List[str] = []
        text = "".join(chunk for _, chunk in printed)
        if text:
            parts.append(text.rstrip("\n"))
        if value is not None:
            parts.append(f"=> {value!r}")
        return "\n".join(parts) if parts else "(no output)"


class MontyReplSandboxFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        monty_repl_sandbox_configuration: MontyReplSandboxConfiguration = (
            MontyReplSandboxConfiguration()
        ),
    ) -> MontyReplSandbox:
        configuration = monty_repl_sandbox_configuration
        return MontyReplSandbox(
            enabled=configuration.enabled,
            workspace_directory=configuration.workspace_directory,
            read_only_mounts=configuration.read_only_mounts,
            max_duration_seconds=configuration.max_duration_seconds,
            max_memory_bytes=configuration.max_memory_bytes,
        )
