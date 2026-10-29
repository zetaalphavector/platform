import asyncio
import functools
import json
import os
import re
from pathlib import Path
from typing import List, Optional

from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.attempt_lineage import (
    Attempt,
    AttemptStore,
    Transcript,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory

_INDEX_FILE = "index.jsonl"
_ATTEMPT_FILE = "attempt.json"
_ARTIFACT_FILE = "artifact.txt"
_TRANSCRIPT_FILE = "transcript.json"
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class FileAttemptStoreConfiguration(BaseModel):
    enabled: bool = Field(False, description="Persist attempts under a directory.")
    directory: str = Field(
        "attempts",
        description=(
            "Root directory. Each task gets a folder named after its rubric holding "
            "index.jsonl plus one folder per attempt with attempt.json, artifact.txt "
            "and transcript.json."
        ),
    )


class FileAttemptStore(AttemptStore):
    """``<directory>/<task>/index.jsonl`` is the append-only lineage of one task;
    each attempt folder holds the artifact as a plain file so the agent (or a
    Monty read-only mount) can read it without going through the SDK."""

    source_name = "file"

    def __init__(self, enabled: bool, directory: str) -> None:
        self.enabled = enabled
        self.__directory = Path(directory)

    @property
    def directory(self) -> Path:
        return self.__directory

    async def list_attempts(self, task: str) -> List[Attempt]:
        return await self.__offload(self.__read_index, task)

    def __read_index(self, task: str) -> List[Attempt]:
        task_dir = self.__task_dir(task)
        index = task_dir / _INDEX_FILE
        if not index.exists():
            return []
        attempts: List[Attempt] = []
        for line in index.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                attempt = Attempt(**json.loads(line))
            except Exception as e:
                logger.error(f"Skipping unreadable attempt index line: {e}")
                continue
            attempt.has_transcript = (task_dir / attempt.id / _TRANSCRIPT_FILE).exists()
            attempts.append(attempt)
        return attempts

    async def get(self, task: str, attempt_id: str) -> Optional[Attempt]:
        for attempt in await self.list_attempts(task):
            if attempt.id == attempt_id:
                return attempt
        return None

    async def add(self, task: str, attempt: Attempt) -> Attempt:
        return await self.__offload(self.__write_attempt, task, attempt)

    def __write_attempt(self, task: str, attempt: Attempt) -> Attempt:
        task_dir = self.__task_dir(task)
        attempt_dir = task_dir / attempt.id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        record = json.loads(attempt.json(exclude={"has_transcript"}))
        self.__write_atomic(attempt_dir / _ATTEMPT_FILE, json.dumps(record, indent=2))
        self.__write_atomic(attempt_dir / _ARTIFACT_FILE, attempt.artifact)
        with (task_dir / _INDEX_FILE).open("a") as index:
            index.write(json.dumps(record) + "\n")
        return attempt

    async def save_transcript(
        self, task: str, attempt_id: str, transcript: Transcript
    ) -> None:
        await self.__offload(self.__write_transcript, task, attempt_id, transcript)

    def __write_transcript(
        self, task: str, attempt_id: str, transcript: Transcript
    ) -> None:
        attempt_dir = self.__task_dir(task) / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        self.__write_atomic(
            attempt_dir / _TRANSCRIPT_FILE, json.dumps(transcript, default=str)
        )

    async def load_transcript(self, task: str, attempt_id: str) -> Optional[Transcript]:
        return await self.__offload(self.__read_transcript, task, attempt_id)

    def __read_transcript(self, task: str, attempt_id: str) -> Optional[Transcript]:
        path = self.__task_dir(task) / attempt_id / _TRANSCRIPT_FILE
        if not path.exists():
            return None
        return json.loads(path.read_text())

    @staticmethod
    async def __offload(fn, *args):
        # Disk IO runs in the default executor so the agent loop is not blocked
        # while an attempt or its transcript is written.
        return await asyncio.get_running_loop().run_in_executor(
            None, functools.partial(fn, *args)
        )

    def __task_dir(self, task: str) -> Path:
        return self.__directory / (_UNSAFE_PATH_CHARS.sub("_", task) or "task")

    @staticmethod
    def __write_atomic(path: Path, content: str) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(content)
        os.replace(temp_path, path)


class FileAttemptStoreFactory(AgentDependencyFactory):
    __singleton__ = True

    @classmethod
    def create(
        cls,
        file_attempt_store_configuration: FileAttemptStoreConfiguration = (
            FileAttemptStoreConfiguration()
        ),
    ) -> FileAttemptStore:
        return FileAttemptStore(
            enabled=file_attempt_store_configuration.enabled,
            directory=file_attempt_store_configuration.directory,
        )
