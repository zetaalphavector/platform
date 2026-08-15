import asyncio
import json
import random
from collections import deque
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from typing_extensions import Literal
from zav.llm_tracing import Span
from zav.prompt_completion import ChatClientRequest, ChatCompletionClient
from zav.prompt_completion import ChatMessage as PcChatMessage
from zav.prompt_completion import ChatMessageSender as PcChatMessageSender
from zav.prompt_completion import ChatResponse as PcChatResponse

from zav.agents_sdk.adapters.llm_models.chat_completion_types import (
    ChatCompletion,
    ChatCompletionSender,
    ChatResponse,
    ToolCallRequest,
    ToolCallResponse,
    parse_chat_message,
)
from zav.agents_sdk.adapters.llm_models.chat_request_builder import ChatRequestBuilder
from zav.agents_sdk.adapters.llm_models.chat_turn_runner import ChatTurnRunner
from zav.agents_sdk.adapters.llm_models.context_window_manager import (
    ContextWindowManager,
)
from zav.agents_sdk.adapters.llm_models.zav_chat_completion_client import (
    ResumableZAVChatCompletionClient,
    ZAVChatCompletionClient,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.chat_message import ChatMessage, FunctionCallRequest
from zav.agents_sdk.domain.tools import Tool, ToolsRegistry, ToolStreamingConfig

REPLAY_EXHAUSTED_MESSAGE = (
    "The replay recording is exhausted: every scripted completion has already "
    "been served. Start a new conversation to run the scripted mission again."
)

REPLAY_MISSION_PROMPT = (
    "Run a full research and data analysis mission on retrieval system "
    "performance: search the index and the web for relevant material, read the "
    "key documents, load and analyse the benchmark datasets, produce plots, "
    "delegate deep-dive subtasks where useful, organise the findings with "
    "tags, and finish with a detailed written report."
)

_MODEL_TTFT_RANGE = (1.0, 3.0)
_MODEL_TOOL_DECIDE_RANGE = (0.5, 2.0)
_MODEL_STREAM_RANGE = (2.0, 10.0)
_DEFAULT_TOOL_DURATION_RANGE = (0.3, 1.5)
_TOOL_DURATION_RANGES: Dict[str, Tuple[float, float]] = {
    "task": (45.0, 180.0),
    "load_dataframe": (4.0, 12.0),
    "read_document": (2.0, 5.0),
    "web_fetch": (2.0, 8.0),
    "web_search": (2.0, 6.0),
    "search": (1.0, 2.5),
    "browse": (1.0, 2.5),
    "search_my_documents": (1.0, 2.5),
    "search_in_tag": (1.0, 2.5),
    "locate_document": (1.0, 2.0),
    "create_basic_plot": (1.0, 3.0),
    "create_multi_series_plot": (1.0, 3.0),
    "create_distribution_plot": (1.0, 3.0),
    "create_time_series_plot": (1.0, 3.0),
    "create_correlation_heatmap": (1.0, 3.0),
}

_WORDS = (
    "retrieval",
    "transformer",
    "benchmark",
    "latency",
    "pipeline",
    "embedding",
    "index",
    "throughput",
    "ablation",
    "baseline",
    "corpus",
    "ranking",
    "query",
    "token",
    "context",
    "window",
    "agent",
    "checkpoint",
    "stream",
    "resume",
    "shard",
    "replica",
    "cache",
    "vector",
    "recall",
    "precision",
    "drift",
    "regression",
    "cluster",
    "quantization",
)

_RecordedCall = Tuple[str, Dict[str, Any], str]


def _tool_call_key(name: str, params: Optional[Dict[str, Any]]) -> str:
    return f"{name}:{json.dumps(params or {}, sort_keys=True, default=str)}"


def _sentence(rng: random.Random, min_words: int = 8, max_words: int = 16) -> str:
    words = [rng.choice(_WORDS) for _ in range(rng.randint(min_words, max_words))]
    return words[0].capitalize() + " " + " ".join(words[1:]) + "."


def _paragraph(rng: random.Random, sentences: int = 4) -> str:
    return " ".join(_sentence(rng) for _ in range(sentences))


def _passage(rng: random.Random, paragraphs: int) -> str:
    return "\n\n".join(_paragraph(rng, rng.randint(3, 6)) for _ in range(paragraphs))


def _doc_id(rng: random.Random, step: int) -> str:
    return f"doc-{step:04d}-{rng.randrange(16 ** 6):06x}"


def _hits_json(rng: random.Random, step: int, count: int = 8) -> str:
    return json.dumps(
        {
            "hits": [
                {
                    "guid": _doc_id(rng, step),
                    "title": _sentence(rng, 4, 9),
                    "score": round(rng.uniform(0.2, 0.99), 4),
                    "snippet": _sentence(rng, 12, 20),
                }
                for _ in range(count)
            ],
            "total": rng.randint(count, 5000),
        }
    )


def _intro_calls(rng: random.Random) -> List[List[_RecordedCall]]:
    return [
        [
            (
                "get_index_configuration",
                {},
                json.dumps(
                    {
                        "index": "research-navigator",
                        "retrieval_units": ["document", "chunk"],
                        "fields": rng.randint(30, 60),
                    }
                ),
            )
        ],
        [
            (
                "list_user_tags",
                {},
                json.dumps(
                    {
                        "tags": [
                            {"id": f"tag-{k}", "name": rng.choice(_WORDS)}
                            for k in range(rng.randint(3, 8))
                        ]
                    }
                ),
            )
        ],
        [
            (
                "list_my_documents",
                {"limit": 25},
                json.dumps(
                    {
                        "documents": [
                            {"id": _doc_id(rng, 0), "title": _sentence(rng, 4, 8)}
                            for _ in range(rng.randint(5, 12))
                        ],
                        "total": rng.randint(12, 300),
                    }
                ),
            )
        ],
    ]


def _cycle_calls(rng: random.Random, cycle: int) -> List[List[_RecordedCall]]:
    base_step = cycle * 100
    df = f"df-{cycle:03d}"
    df_b = f"df-{cycle:03d}-b"
    tag = f"tag-mission-{cycle}"
    document_id = _doc_id(rng, base_step + 4)
    plot_uri = f"s3://perf-plots/mission-{cycle}/basic.png"
    return [
        [
            (
                "search",
                {
                    "query": f"{_sentence(rng, 4, 7)} (pass {cycle})",
                    "retrieval_unit": "document",
                },
                _hits_json(rng, base_step + 1),
            )
        ],
        [
            (
                "get_filter_options",
                {"field": f"facet_{cycle}"},
                json.dumps({"values": [rng.choice(_WORDS) for _ in range(6)]}),
            ),
            (
                "get_document_fields",
                {"prefix": f"f{cycle}"},
                json.dumps({"fields": [f"field_{cycle}_{k}" for k in range(8)]}),
            ),
        ],
        [
            (
                "browse",
                {"filters": {"year": 2018 + cycle % 8}, "page": cycle},
                _hits_json(rng, base_step + 3),
            )
        ],
        [
            (
                "read_document",
                {"document_id": document_id},
                _passage(rng, rng.randint(6, 10)),
            )
        ],
        [
            (
                "retrieve_metadata",
                {"document_id": document_id},
                json.dumps(
                    {
                        "title": _sentence(rng, 5, 10),
                        "authors": [rng.choice(_WORDS) for _ in range(3)],
                        "year": 2018 + cycle % 8,
                        "venue": rng.choice(_WORDS),
                    }
                ),
            )
        ],
        [
            (
                "locate_document",
                {"title": _sentence(rng, 5, 9)},
                json.dumps({"document_id": _doc_id(rng, base_step + 6)}),
            )
        ],
        [
            (
                "web_search",
                {"query": f"{_sentence(rng, 4, 7)} (web pass {cycle})"},
                _hits_json(rng, base_step + 7, count=6),
            ),
            (
                "search_my_documents",
                {"query": f"{_sentence(rng, 4, 7)} (mine pass {cycle})"},
                _hits_json(rng, base_step + 7, count=4),
            ),
        ],
        [
            (
                "web_fetch",
                {
                    "url": (
                        f"https://example.org/reports/{cycle}"
                        f"/{rng.randrange(10 ** 8)}"
                    )
                },
                _passage(rng, rng.randint(5, 9)),
            )
        ],
        [
            (
                "load_dataframe",
                {"source": f"s3://perf-bucket/data-{cycle}.parquet"},
                json.dumps(
                    {
                        "dataframe_id": df,
                        "rows": rng.randint(10_000, 900_000),
                        "columns": [f"metric_{cycle}", f"category_{cycle}", "ts"],
                    }
                ),
            )
        ],
        [
            (
                "get_dataframe_schema",
                {"dataframe_id": df},
                json.dumps(
                    {
                        "schema": {
                            f"metric_{cycle}": "float64",
                            f"category_{cycle}": "string",
                            "ts": "datetime64[ns]",
                        }
                    }
                ),
            )
        ],
        [
            (
                "preview_dataframe",
                {"dataframe_id": df, "rows": 10},
                json.dumps(
                    {
                        "rows": [
                            {
                                f"metric_{cycle}": round(rng.uniform(0, 100), 3),
                                f"category_{cycle}": rng.choice(_WORDS),
                            }
                            for _ in range(10)
                        ]
                    }
                ),
            )
        ],
        [
            (
                "count_dataframe_rows",
                {"dataframe_id": df},
                json.dumps({"count": rng.randint(10_000, 900_000)}),
            ),
            (
                "get_missing_value_counts",
                {"dataframe_id": df},
                json.dumps({f"metric_{cycle}": rng.randint(0, 500), "ts": 0}),
            ),
            (
                "get_column_statistics",
                {"dataframe_id": df, "column": f"metric_{cycle}"},
                json.dumps(
                    {
                        "mean": round(rng.uniform(10, 90), 4),
                        "std": round(rng.uniform(1, 20), 4),
                        "min": 0.0,
                        "max": round(rng.uniform(90, 200), 4),
                    }
                ),
            ),
        ],
        [
            (
                "get_column_values",
                {"dataframe_id": df, "column": f"category_{cycle}"},
                json.dumps({"values": [rng.choice(_WORDS) for _ in range(12)]}),
            )
        ],
        [
            (
                "filter_dataframe_rows",
                {
                    "dataframe_id": df,
                    "condition": f"metric_{cycle} > {rng.randint(10, 60)}",
                },
                json.dumps({"dataframe_id": df_b, "rows": rng.randint(500, 90_000)}),
            )
        ],
        [
            (
                "sort_dataframe_rows",
                {"dataframe_id": df_b, "by": f"metric_{cycle}", "ascending": False},
                json.dumps({"dataframe_id": df_b, "sorted_by": f"metric_{cycle}"}),
            )
        ],
        [
            (
                "aggregate_column",
                {
                    "dataframe_id": df_b,
                    "column": f"metric_{cycle}",
                    "operation": "mean",
                },
                json.dumps({"value": round(rng.uniform(20, 80), 4)}),
            )
        ],
        [
            (
                "read_dataframe",
                {"dataframe_id": df_b, "offset": cycle},
                json.dumps(
                    {
                        "rows": [
                            {f"metric_{cycle}": round(rng.uniform(0, 100), 3)}
                            for _ in range(20)
                        ]
                    }
                ),
            )
        ],
        [
            (
                "merge_dataframes",
                {"left": df, "right": df_b, "on": f"key_{cycle}"},
                json.dumps(
                    {"dataframe_id": f"{df}-merged", "rows": rng.randint(1, 10**5)}
                ),
            )
        ],
        [
            (
                "concatenate_dataframes",
                {"dataframe_ids": [df, df_b]},
                json.dumps(
                    {"dataframe_id": f"{df}-concat", "rows": rng.randint(1, 10**6)}
                ),
            )
        ],
        [
            (
                "create_basic_plot",
                {
                    "dataframe_id": df_b,
                    "x": "ts",
                    "y": f"metric_{cycle}",
                    "kind": "line",
                },
                json.dumps({"image_uri": plot_uri, "caption": _sentence(rng, 5, 9)}),
            )
        ],
        [
            (
                "create_multi_series_plot",
                {"dataframe_id": df_b, "x": "ts", "series": [f"metric_{cycle}"]},
                json.dumps(
                    {
                        "image_uri": f"s3://perf-plots/mission-{cycle}/multi.png",
                        "caption": _sentence(rng, 5, 9),
                    }
                ),
            )
        ],
        [
            (
                "create_distribution_plot",
                {"dataframe_id": df_b, "column": f"metric_{cycle}"},
                json.dumps(
                    {
                        "image_uri": f"s3://perf-plots/mission-{cycle}/dist.png",
                        "caption": _sentence(rng, 5, 9),
                    }
                ),
            )
        ],
        [
            (
                "create_time_series_plot",
                {"dataframe_id": df_b, "x": "ts", "y": f"metric_{cycle}"},
                json.dumps(
                    {
                        "image_uri": f"s3://perf-plots/mission-{cycle}/ts.png",
                        "caption": _sentence(rng, 5, 9),
                    }
                ),
            )
        ],
        [
            (
                "create_correlation_heatmap",
                {"dataframe_id": df_b, "columns": [f"metric_{cycle}", "ts"]},
                json.dumps(
                    {
                        "image_uri": f"s3://perf-plots/mission-{cycle}/corr.png",
                        "caption": _sentence(rng, 5, 9),
                    }
                ),
            )
        ],
        [
            (
                "validate_image_hit",
                {"image_uri": plot_uri},
                json.dumps({"valid": True, "reason": _sentence(rng, 5, 9)}),
            )
        ],
        [
            (
                "task",
                {
                    "description": f"(mission {cycle}) {_paragraph(rng, 2)}",
                    "agent": "zav_orchestrator",
                },
                _passage(rng, rng.randint(4, 7)),
            )
        ],
        [
            (
                "create_tag",
                {"name": tag},
                json.dumps({"tag_id": tag}),
            )
        ],
        [
            (
                "tag_document",
                {"tag_id": tag, "document_id": document_id},
                json.dumps({"status": "tagged"}),
            )
        ],
        [
            (
                "search_in_tag",
                {"tag_id": tag, "query": f"{_sentence(rng, 4, 7)} (tag pass {cycle})"},
                _hits_json(rng, base_step + 29, count=3),
            )
        ],
        [
            (
                "list_documents_in_tag",
                {"tag_id": tag},
                json.dumps({"documents": [document_id]}),
            )
        ],
        [
            (
                "untag_document",
                {"tag_id": tag, "document_id": document_id},
                json.dumps({"status": "untagged"}),
            )
        ],
    ]


def _final_answer(rng: random.Random) -> str:
    findings = "\n".join(
        f"- {_sentence(rng, 10, 18)}" for _ in range(rng.randint(6, 10))
    )
    table_rows = "\n".join(
        f"| metric_{k} | {round(rng.uniform(10, 90), 2)} | "
        f"{round(rng.uniform(0.1, 0.99), 3)} | {rng.choice(_WORDS)} |"
        for k in range(5)
    )
    sources = "\n".join(
        f"{k + 1}. {_doc_id(rng, k)} — {_sentence(rng, 4, 8)}" for k in range(6)
    )
    return "\n\n".join(
        [
            "## Mission Report\n\n" + _paragraph(rng, 3),
            "### Key Findings\n\n" + findings,
            "### Data Summary\n\n| column | mean | score | label |\n"
            "| --- | --- | --- | --- |\n" + table_rows,
            "### Sources\n\n" + sources,
            "### Conclusion\n\n" + _paragraph(rng, 4),
        ]
    )


def generate_synthetic_completions(
    steps: int = 40, seed: int = 7
) -> List[ChatCompletion]:
    """Builds a scripted transcript shaped exactly like a persisted resumable
    chat state dump: one user request, ``steps`` model/tool round-trips over
    the tool inventory of a production agent, and a long final answer."""
    rng = random.Random(seed)
    step_batches: List[List[_RecordedCall]] = list(_intro_calls(rng))
    cycle = 0
    while len(step_batches) < max(1, steps):
        step_batches.extend(_cycle_calls(rng, cycle))
        cycle += 1
    step_batches = step_batches[: max(1, steps)]

    completions = [
        ChatCompletion(sender=ChatCompletionSender.USER, content=REPLAY_MISSION_PROMPT)
    ]
    call_counter = 0
    for batch in step_batches:
        requests: List[ToolCallRequest] = []
        responses: List[ToolCallResponse] = []
        for name, params, response in batch:
            call_id = f"call_{call_counter:05d}"
            call_counter += 1
            requests.append(
                ToolCallRequest(
                    id=call_id,
                    function_call_request=FunctionCallRequest(name=name, params=params),
                )
            )
            responses.append(ToolCallResponse(id=call_id, response=response))
        completions.append(
            ChatCompletion(
                sender=ChatCompletionSender.BOT,
                content="",
                tool_call_requests=requests,
            )
        )
        completions.append(
            ChatCompletion(
                sender=ChatCompletionSender.TOOL,
                content="",
                tool_call_responses=responses,
            )
        )
    completions.append(
        ChatCompletion(sender=ChatCompletionSender.BOT, content=_final_answer(rng))
    )
    return completions


class ReplayScript:
    """A recorded transcript indexed for replay.

    ``bot_completions`` is the completion script the model stub serves in
    order, and ``tool_responses`` maps each recorded tool call (by name plus
    canonical params) to its recorded outputs in call order. Built from the
    same ``chat_completions`` shape the resumable client dumps, so a synthetic
    recording and a persisted real one are interchangeable inputs.
    """

    def __init__(self, completions: List[ChatCompletion]) -> None:
        self.completions = list(completions)
        self.bot_completions = [
            completion
            for completion in self.completions
            if completion.sender == ChatCompletionSender.BOT
        ]
        self.tool_responses: Dict[str, Deque[str]] = {}
        self.tool_names: Set[str] = set()
        request_keys: Dict[str, str] = {}
        for completion in self.completions:
            if completion.sender == ChatCompletionSender.BOT:
                for request in completion.tool_call_requests or []:
                    name = request.function_call_request.name
                    self.tool_names.add(name)
                    request_keys[request.id] = _tool_call_key(
                        name, request.function_call_request.params
                    )
            elif completion.sender == ChatCompletionSender.TOOL:
                for response in completion.tool_call_responses or []:
                    key = request_keys.get(response.id)
                    if key is not None:
                        self.tool_responses.setdefault(key, deque()).append(
                            response.response or ""
                        )


class ReplayChatCompletionClient:
    """Serves recorded model completions instead of calling an LLM vendor.

    The cursor is stateless: the next completion is the recorded BOT
    completion at the index equal to the number of BOT messages already in the
    incoming request's conversation. A fresh turn starts at zero and a turn
    restored from a checkpoint on another pod lands on the right step without
    any replay-side persistence. Compaction only rewrites message content, so
    the BOT count survives context window management.
    """

    def __init__(
        self, script: ReplayScript, time_scale: float = 1.0, seed: int = 7
    ) -> None:
        self.__script = script
        self.__time_scale = time_scale
        self.__seed = seed

    async def complete(
        self, request: ChatClientRequest, stream: bool = False
    ) -> Union[AsyncIterator[PcChatResponse], PcChatResponse]:
        position = sum(
            1
            for message in request["conversation"].messages
            if message.sender == PcChatMessageSender.BOT
        )
        message = self.__message_at(position)
        ttft, body_seconds = self.__timings(position, message)
        if stream:
            return self.__stream(message, ttft, body_seconds)
        await self.__sleep(ttft + body_seconds)
        return PcChatResponse(error=None, chat_message=message)

    def __message_at(self, position: int) -> PcChatMessage:
        bot_completions = self.__script.bot_completions
        if position < len(bot_completions):
            return parse_chat_message(bot_completions[position])
        return PcChatMessage(
            sender=PcChatMessageSender.BOT, content=REPLAY_EXHAUSTED_MESSAGE
        )

    def __timings(self, position: int, message: PcChatMessage) -> Tuple[float, float]:
        rng = random.Random(f"{self.__seed}:model:{position}")
        ttft = rng.uniform(*_MODEL_TTFT_RANGE)
        if message.tool_call_requests or not message.content:
            return ttft, rng.uniform(*_MODEL_TOOL_DECIDE_RANGE)
        return ttft, rng.uniform(*_MODEL_STREAM_RANGE)

    async def __stream(
        self, message: PcChatMessage, ttft: float, body_seconds: float
    ) -> AsyncIterator[PcChatResponse]:
        await self.__sleep(ttft)
        if message.tool_call_requests or not message.content:
            await self.__sleep(body_seconds)
            yield PcChatResponse(error=None, chat_message=message)
            return

        words = message.content.split(" ")
        chunk_size = 6
        chunk_count = max(1, (len(words) + chunk_size - 1) // chunk_size)
        delay = body_seconds / chunk_count
        for end in range(chunk_size, len(words), chunk_size):
            await self.__sleep(delay)
            yield PcChatResponse(
                error=None,
                chat_message=PcChatMessage(
                    sender=message.sender, content=" ".join(words[:end])
                ),
            )
        await self.__sleep(delay)
        yield PcChatResponse(error=None, chat_message=message)

    async def __sleep(self, seconds: float) -> None:
        scaled = seconds * self.__time_scale
        if scaled > 0:
            await asyncio.sleep(scaled)


def _stand_in_executable(**params: Any) -> str:
    raise RuntimeError(
        "Replay stand-in tools must be served from the recording, never executed."
    )


class ReplayToolsRegistry(ToolsRegistry):
    """Serves recorded tool responses instead of running tool executables.

    The wrapped agent registry is copied so streaming display config keeps
    producing real tool progress events, but ``llm_response_transform`` is
    stripped because recorded responses are already post-transform. Recorded
    tools missing from the agent registry get stand-in entries so the scripted
    loop works without any live tool providers. Tools added later by the
    framework (e.g. context window retrieval tools) still execute for real.
    """

    def __init__(
        self,
        script: ReplayScript,
        source: Optional[ToolsRegistry] = None,
        time_scale: float = 1.0,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.__responses = {
            key: deque(responses) for key, responses in script.tool_responses.items()
        }
        self.__time_scale = time_scale
        self.__seed = seed
        if source is not None:
            for tool in source.tools_index.values():
                self.tools_index[tool.name] = self.__without_llm_transform(tool)
        for name in sorted(script.tool_names):
            if name not in self.tools_index:
                self.tools_index[name] = self.__stand_in_tool(name)
        self.__replay_names = set(self.tools_index)

    async def execute(self, name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        key = _tool_call_key(name, params)
        queue = self.__responses.get(key)
        if queue:
            await self.__sleep(name, key)
            return queue.popleft()
        if name in self.tools_index and name not in self.__replay_names:
            return await super().execute(name=name, params=params)
        raise ValueError(
            f"Replay recording has no response for tool '{name}' with params "
            f"{json.dumps(params or {}, sort_keys=True, default=str)}"
        )

    async def __sleep(self, name: str, key: str) -> None:
        low, high = _TOOL_DURATION_RANGES.get(name, _DEFAULT_TOOL_DURATION_RANGE)
        rng = random.Random(f"{self.__seed}:tool:{key}")
        scaled = rng.uniform(low, high) * self.__time_scale
        if scaled > 0:
            await asyncio.sleep(scaled)

    def __without_llm_transform(self, tool: Tool) -> Tool:
        streaming_config = tool.streaming_config
        if streaming_config is None or streaming_config.llm_response_transform is None:
            return tool
        return tool.copy(
            update={
                "streaming_config": streaming_config.copy(
                    update={"llm_response_transform": None}
                )
            }
        )

    def __stand_in_tool(self, name: str) -> Tool:
        return Tool(
            name=name,
            description=f"Replay stand-in for recorded tool '{name}'.",
            executable=_stand_in_executable,
            parameters_spec={"type": "object", "properties": {}},
            streaming_config=ToolStreamingConfig(
                running_text=f"Running {name}",
                completed_text=f"Finished {name}",
            ),
        )


class ReplayResumableZAVChatCompletionClient(ResumableZAVChatCompletionClient):
    """A resumable client whose tool execution is also served from the script.

    Everything else — turn state, checkpoint dump/load, streaming, context
    window management — is the real implementation.
    """

    def __init__(
        self,
        client: ZAVChatCompletionClient,
        runner: ChatTurnRunner,
        script: ReplayScript,
        time_scale: float = 1.0,
        seed: int = 7,
    ) -> None:
        super().__init__(client, runner)
        self.__script = script
        self.__time_scale = time_scale
        self.__seed = seed

    async def complete(
        self,
        messages: Optional[List[ChatMessage]] = None,
        completions: Optional[List[ChatCompletion]] = None,
        max_tokens: Optional[int] = None,
        bot_setup_description: Optional[str] = None,
        functions: Optional[List[Dict]] = None,
        tools: Optional[Union[ToolsRegistry, List[Dict]]] = None,
        tool_choice: Optional[str] = None,
        stream: Union[Literal[True, False], bool] = False,
        execute_tools: bool = True,
        stream_tool_calls: bool = False,
        stream_tool_events: Optional[bool] = None,
        stream_tool_progress: bool = False,
        preserve_streamed_content_parts: bool = False,
        log_fn: Optional[Callable] = None,
        max_nesting_level: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        logprobs: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        concurrent_tool_execution: bool = False,
        seed: Optional[int] = None,
        verbosity: Optional[str] = None,
    ) -> Union[AsyncIterator[ChatResponse], ChatResponse]:
        if not isinstance(tools, ReplayToolsRegistry) and (
            tools is None or isinstance(tools, ToolsRegistry)
        ):
            tools = ReplayToolsRegistry(
                script=self.__script,
                source=tools,
                time_scale=self.__time_scale,
                seed=self.__seed,
            )
        return await super().complete(
            messages=messages,
            completions=completions,
            max_tokens=max_tokens,
            bot_setup_description=bot_setup_description,
            functions=functions,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            execute_tools=execute_tools,
            stream_tool_calls=stream_tool_calls,
            stream_tool_events=stream_tool_events,
            stream_tool_progress=stream_tool_progress,
            preserve_streamed_content_parts=preserve_streamed_content_parts,
            log_fn=log_fn,
            max_nesting_level=max_nesting_level,
            reasoning_effort=reasoning_effort,
            logprobs=logprobs,
            parallel_tool_calls=parallel_tool_calls,
            concurrent_tool_execution=concurrent_tool_execution,
            seed=seed,
            verbosity=verbosity,
        )


class ReplayResumableZAVChatCompletionClientFactory(AgentDependencyFactory):
    """Builds a resumable client that replays a synthetic recording.

    Registered under the ``ReplayResumableZAVChatCompletionClient`` subclass,
    so only agents that declare that exact dependency type get scripted
    completions — every other agent on the same deployment keeps the real
    factory. The replay knobs are defaults, so an agent setup can tune
    ``replay_steps``, ``replay_seed`` and ``replay_time_scale`` through its
    agent configuration without a redeploy; replicas sharing a setup replay
    the same recording. ``max_nesting_level`` is raised to fit the script so
    long missions never hit the final-turn cutoff, and no vendor client is
    ever built, so no API keys are needed.
    """

    @classmethod
    def create(
        cls,
        context_window_manager: ContextWindowManager,
        max_nesting_level: int = 20,
        replay_steps: int = 40,
        replay_seed: int = 7,
        replay_time_scale: float = 1.0,
        span: Optional[Span] = None,
    ) -> ReplayResumableZAVChatCompletionClient:
        script = ReplayScript(
            generate_synthetic_completions(steps=replay_steps, seed=replay_seed)
        )
        replay_chat_client = cast(
            ChatCompletionClient,
            ReplayChatCompletionClient(
                script=script,
                time_scale=replay_time_scale,
                seed=replay_seed,
            ),
        )
        nesting_level = max(max_nesting_level, len(script.bot_completions) + 1)
        client = ZAVChatCompletionClient(
            replay_chat_client,
            context_window_manager=context_window_manager,
            max_nesting_level=nesting_level,
            span=span,
        )
        runner = ChatTurnRunner(
            chat_completion_client=replay_chat_client,
            request_builder=ChatRequestBuilder(context_window_manager),
            span=span,
        )
        return ReplayResumableZAVChatCompletionClient(
            client,
            runner,
            script=script,
            time_scale=replay_time_scale,
            seed=replay_seed,
        )
