from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, List, Literal, Optional, Sequence

from zav.llm_tracing import LocalTraceStore
from zav.logging import logger

from zav.agents_sdk.behavior.assertions import AssertionRegistry
from zav.agents_sdk.behavior.models import TestSpecification, load_spec
from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
from zav.agents_sdk.domain.chat_message import (
    ChatMessage,
    ChatMessageSender,
    ConversationContext,
)


@dataclass
class TestCaseResult:
    """Outcome for a single behavior spec execution."""

    spec_id: str
    description: Optional[str]
    status: Literal["passed", "failed", "running"]
    error: Optional[str] = None
    path: Path | None = None
    duration: float | None = None  # seconds


class TestHarness:
    """Light-weight orchestrator around test specifications."""

    def __init__(
        self, trace_store: LocalTraceStore, spec_paths: Sequence[Path] | None = None
    ):
        self.__spec_paths: List[Path] = (
            [Path(p) for p in spec_paths] if spec_paths else []
        )
        self.__specs: List[TestSpecification] = []
        self.__trace_store = trace_store

    def discover(self, root: Path | str, pattern: str = "*.yaml"):
        """Recursively collect YAML spec files under *root* matching *pattern*."""
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(root)
        for path in root.rglob(pattern):
            self.__spec_paths.append(path)

    def add_spec(self, spec_path: Path):
        """Add a spec file to the harness."""
        self.__spec_paths.append(spec_path)

    def load(self):
        """Parse previously discovered spec files into Pydantic models."""
        if not self.__spec_paths:
            logger.warning("No spec paths provided – nothing to load")
            return
        for path in self.__spec_paths:
            try:
                spec = load_spec(path)
                self.__specs.append(spec)
            except Exception as exc:
                logger.error("Failed to load spec %s: %s", path, exc)
                raise

    async def run_iter(
        self, agent_factory: ChatAgentFactory
    ) -> AsyncIterator[TestCaseResult]:
        """Async iterator yielding TestCaseResult with status 'running' before,
        then pass/fail after."""

        if not self.__specs:
            raise RuntimeError("No specs loaded – call .load() first")

        for idx, spec in enumerate(self.__specs):
            spec_path: Path | None = (
                self.__spec_paths[idx] if idx < len(self.__spec_paths) else None
            )
            # Yield a running event so CLI can print the spec info before running
            yield TestCaseResult(
                spec_id=spec.id,
                description=spec.description,
                status="running",
                path=spec_path,
            )
            conversation: List[ChatMessage] = [
                ChatMessage(
                    sender=ChatMessageSender(message.role),
                    content=message.content,
                )
                for message in spec.messages
            ]
            handler_params = spec.bot_params or {}
            agent = await agent_factory.create(
                agent_identifier=spec.agent_identifier,
                handler_params={
                    "request_headers": {},
                    **handler_params,
                },
                conversation_context=(
                    ConversationContext(
                        document_context=spec.conversation_context.document_context,
                        custom_context=spec.conversation_context.custom_context,
                    )
                    if spec.conversation_context
                    else None
                ),
            )

            start_time = time.perf_counter()
            response = await agent.execute(conversation)
            duration = time.perf_counter() - start_time

            if response is None:
                yield TestCaseResult(
                    spec_id=spec.id,
                    description=spec.description,
                    status="failed",
                    error="No response returned by agent.",
                    path=spec_path,
                    duration=duration,
                )
                continue

            try:
                AssertionRegistry.evaluate_response(
                    self.__trace_store, response, spec.expectations
                )
            except AssertionError as exc:
                yield TestCaseResult(
                    spec_id=spec.id,
                    description=spec.description,
                    status="failed",
                    error=str(exc),
                    path=spec_path,
                    duration=duration,
                )
                continue

            yield TestCaseResult(
                spec_id=spec.id,
                description=spec.description,
                status="passed",
                path=spec_path,
                duration=duration,
            )

    async def run_async(self, agent_factory: ChatAgentFactory) -> List[TestCaseResult]:
        """Execute all loaded specs asynchronously and return collected results."""
        return [result async for result in self.run_iter(agent_factory)]

    def run(self, agent_factory: ChatAgentFactory) -> List[TestCaseResult]:
        """Execute all loaded specs synchronously via asyncio.run."""
        return asyncio.run(self.run_async(agent_factory))


__all__ = ["TestHarness", "TestCaseResult"]
