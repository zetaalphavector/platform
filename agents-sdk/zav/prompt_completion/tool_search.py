import json
import re
from collections import OrderedDict
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union, cast

from zav.llm_domain import ToolSearchCatalogue, ToolSearchMethodSettings, ToolSearchMode
from zav.logging import logger

from zav.prompt_completion.client import (
    BotConversation,
    ChatClientRequest,
    ChatCompletionClient,
    ChatMessage,
    ChatMessageSender,
    ChatResponse,
    ToolCallRequest,
    ToolCallResponse,
)

SEARCH_TOOLS_NAME = "search_tools"
# A model that keeps searching instead of answering gets the tools it found
# after this many rounds and must use them.
MAX_SEARCH_ROUNDS = 3


def is_deferred(tool: Dict[str, Any]) -> bool:
    return bool(tool.get("defer_loading"))


def tool_name(tool: Dict[str, Any]) -> str:
    return cast(str, tool["function"]["name"])


def _without_deferral(tool: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in tool.items()
        if key not in ("defer_loading", "namespace")
    }


def _loaded(tool: Dict[str, Any]) -> Dict[str, Any]:
    """A deferred tool sent up front, still in its namespace."""
    return {key: value for key, value in tool.items() if key != "defer_loading"}


def namespaces_of(
    tools: List[Dict[str, Any]],
) -> "OrderedDict[str, Tuple[str, List[Dict[str, Any]]]]":
    """Tools grouped by namespace name, in declaration order; a tool without a
    namespace is its own group."""
    groups: "OrderedDict[str, Tuple[str, List[Dict[str, Any]]]]" = OrderedDict()
    for tool in tools:
        namespace = tool.get("namespace") or {}
        name = namespace.get("name") or tool_name(tool)
        description = namespace.get("description") or tool["function"].get(
            "description", ""
        )
        groups.setdefault(name, (description, []))[1].append(tool)
    return groups


def _called_tool_names(conversation: BotConversation) -> Set[str]:
    names: Set[str] = set()
    for message in conversation.messages:
        if message.function_call_request is not None:
            names.add(message.function_call_request.function_name)
        for request in message.tool_call_requests or []:
            names.add(request.function_call_request.function_name)
    return names


def search_call_ids(conversation: BotConversation) -> Set[str]:
    return {
        request.id
        for message in conversation.messages
        for request in message.tool_call_requests or []
        if request.function_call_request.function_name == SEARCH_TOOLS_NAME
    }


def found_tool_names(tool_response: Optional[str]) -> List[str]:
    """The tools a `search_tools` answer loaded."""
    try:
        answer = json.loads(tool_response or "{}")
    except json.JSONDecodeError:
        return []
    return list(answer.get("loaded") or []) if isinstance(answer, dict) else []


def _search_deferred_tools(
    deferred: List[Dict[str, Any]],
    names: Optional[List[str]] = None,
    query: Optional[str] = None,
    *,
    max_results: int,
) -> List[Dict[str, Any]]:
    """Exact names (a tool or a whole namespace) first, then up to `max_results`
    tools ranked by how many query words their name, namespace and description
    contain."""
    groups = namespaces_of(deferred)
    by_name = {tool_name(tool): tool for tool in deferred}
    found: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for name in names or []:
        if name in by_name:
            found[name] = by_name[name]
        elif name in groups:
            for tool in groups[name][1]:
                found[tool_name(tool)] = tool
    words = {word for word in re.findall(r"[a-z0-9]+", (query or "").lower())}
    if words:
        scored = []
        for tool in deferred:
            namespace = tool.get("namespace") or {}
            haystack = " ".join(
                [
                    tool_name(tool),
                    tool["function"].get("description", ""),
                    namespace.get("name", ""),
                    namespace.get("description", ""),
                ]
            ).lower()
            score = sum(1 for word in words if word in haystack)
            if score:
                scored.append((score, tool))
        scored.sort(key=lambda item: -item[0])
        for _, tool in scored[:max_results]:
            found.setdefault(tool_name(tool), tool)
    return list(found.values())


def _catalogue_of(
    deferred: List[Dict[str, Any]], catalogue: ToolSearchCatalogue
) -> str:
    lines = []
    for namespace, (description, tools) in namespaces_of(deferred).items():
        line = f"- {namespace}: {description}"
        if catalogue == ToolSearchCatalogue.NAMES:
            line += " Tools: " + ", ".join(tool_name(tool) for tool in tools)
        lines.append(line)
        if catalogue == ToolSearchCatalogue.DESCRIPTIONS:
            lines.extend(
                f"  - {tool_name(tool)}: "
                + (tool["function"].get("description") or "").strip().split("\n")[0]
                for tool in tools
            )
    return "\n".join(lines) or "Every tool is loaded."


def _search_tools_spec(
    deferred: List[Dict[str, Any]],
    catalogue: ToolSearchCatalogue = ToolSearchCatalogue.NAMES,
) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": SEARCH_TOOLS_NAME,
            "description": (
                "Load tools before calling them. These tools exist but are not "
                "loaded yet; pass tool or group names, or describe what you need, "
                "and call the loaded tools in your next step. Call this alone.\n"
                f"{_catalogue_of(deferred, catalogue)}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool or group names to load.",
                    },
                    "query": {
                        "type": "string",
                        "description": "What the tools should do.",
                    },
                },
            },
        },
    }


class ToolSearchChatClient(ChatCompletionClient):
    """Serves a request's deferred tools with the first of its tool search
    methods the model supports. `server` leaves the search to the vendor. `client`
    and `emulated` run our search: a `search_tools` tool lists the deferred tools
    the conversation has not called, and this client answers its calls and asks
    the model again. With `client` the vendor adapter encodes the deferred tools
    and the answers in the vendor's protocol (`defer_loading`, tool references);
    with `emulated` found tools are simply added. A tool the conversation called
    counts as loaded and goes up front. Without a supported method every tool
    goes up front."""

    def __init__(
        self,
        client: ChatCompletionClient,
        methods: List[ToolSearchMode],
        model_name: str,
    ):
        self.__client = client
        self.__methods = methods
        self.__model_name = model_name

    @classmethod
    def from_configuration(cls, vendor_configuration, model_configuration, span=None):
        raise NotImplementedError

    async def complete(  # type: ignore[override]
        self, request: ChatClientRequest, stream: bool = False
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        preferences = request.get("tool_search") or []
        request = cast(
            ChatClientRequest,
            {key: value for key, value in request.items() if key != "tool_search"},
        )
        tools = request.get("tools") or []
        if not any(is_deferred(tool) for tool in tools):
            return await self.__client.complete(request, stream=stream)
        settings = next(
            (s for s in preferences if s.method in self.__methods),
            None,
        )
        if settings is None:
            logger.warning(
                f"No enabled tool search method fits {self.__model_name} "
                f"(enabled: {[s.method.value for s in preferences]}, supported: "
                f"{[m.value for m in self.__methods]}); deferred tools go up front."
            )
            return await self.__client.complete(
                {**request, "tools": [_without_deferral(tool) for tool in tools]},
                stream=stream,
            )
        if settings.method == ToolSearchMode.SERVER:
            return await self.__client.complete(
                {**request, "tool_search_method": ToolSearchMode.SERVER},
                stream=stream,
            )
        if stream:
            return self.__search_streaming(request, settings)
        return await self.__search_once(request, settings)

    def __round_request(
        self,
        request: ChatClientRequest,
        conversation: BotConversation,
        settings: ToolSearchMethodSettings,
        found: Set[str],
        last_round: bool,
    ) -> ChatClientRequest:
        """The request for one round. The last round sends the original
        conversation with the found tools up front and no search tool."""
        called = _called_tool_names(request["conversation"])
        tools = request.get("tools") or []
        eager = [_without_deferral(t) for t in tools if not is_deferred(t)]
        deferred = [t for t in tools if is_deferred(t)]
        native = settings.method == ToolSearchMode.CLIENT
        up_front = _loaded if native else _without_deferral
        promoted = [up_front(t) for t in deferred if tool_name(t) in called]
        if last_round:
            final: ChatClientRequest = {
                **request,
                "tools": [
                    *eager,
                    *promoted,
                    *(up_front(t) for t in deferred if tool_name(t) in found),
                ],
            }
            if native:
                final["tool_search_method"] = ToolSearchMode.CLIENT
            return final
        pending = [t for t in deferred if tool_name(t) not in called | found]
        searchable = [
            *(t for t in deferred if tool_name(t) not in called and native),
            *(
                _without_deferral(t)
                for t in deferred
                if tool_name(t) in found and not native
            ),
        ]
        prepared: ChatClientRequest = {
            **request,
            "conversation": conversation,
            "tools": [
                *eager,
                *promoted,
                *searchable,
                _search_tools_spec(pending, settings.catalogue),
            ],
        }
        if native:
            prepared["tool_search_method"] = ToolSearchMode.CLIENT
        return prepared

    @staticmethod
    def __search_calls(response: ChatResponse) -> List[ToolCallRequest]:
        message = response.chat_message
        if response.error is not None or message is None:
            return []
        return [
            call
            for call in message.tool_call_requests or []
            if call.function_call_request.function_name == SEARCH_TOOLS_NAME
        ]

    @staticmethod
    def __answer_searches(
        request: ChatClientRequest,
        conversation: BotConversation,
        settings: ToolSearchMethodSettings,
        response: ChatResponse,
        found: Set[str],
    ) -> BotConversation:
        """The conversation with the searches and their answers appended; other
        calls in the same response are dropped, the model makes them again once
        the tools are loaded. The found tools join `found`."""
        message = cast(ChatMessage, response.chat_message)
        deferred = [t for t in request.get("tools") or [] if is_deferred(t)]
        searches = [
            call
            for call in message.tool_call_requests or []
            if call.function_call_request.function_name == SEARCH_TOOLS_NAME
        ]
        answers = []
        for call in searches:
            params = call.function_call_request.function_params or {}
            matches = _search_deferred_tools(
                deferred,
                names=params.get("names"),
                query=params.get("query"),
                max_results=settings.max_results,
            )
            found.update(tool_name(tool) for tool in matches)
            answers.append(
                ToolCallResponse(
                    id=call.id,
                    tool_response=json.dumps(
                        {
                            "loaded": [tool_name(tool) for tool in matches],
                            **(
                                {}
                                if matches
                                else {"message": "No tool matched; try other words."}
                            ),
                        }
                    ),
                )
            )
        return BotConversation(
            bot_setup_description=conversation.bot_setup_description,
            messages=[
                *conversation.messages,
                message.copy(
                    update={
                        "tool_call_requests": searches,
                        "response_replay_items": None,
                    }
                ),
                ChatMessage(
                    sender=ChatMessageSender.TOOL,
                    content="",
                    tool_call_responses=answers,
                ),
            ],
        )

    async def __search_once(
        self, request: ChatClientRequest, settings: ToolSearchMethodSettings
    ) -> ChatResponse:
        conversation = request["conversation"]
        found: Set[str] = set()
        # Text written before a search stays in front of the answer.
        carried = ""
        for search_round in range(MAX_SEARCH_ROUNDS + 1):
            response = cast(
                ChatResponse,
                await self.__client.complete(
                    self.__round_request(
                        request,
                        conversation,
                        settings,
                        found,
                        search_round == MAX_SEARCH_ROUNDS,
                    ),
                    stream=False,
                ),
            )
            if not self.__search_calls(response):
                break
            carried += cast(ChatMessage, response.chat_message).content
            conversation = self.__answer_searches(
                request, conversation, settings, response, found
            )
        if carried and response.chat_message is not None:
            response.chat_message.content = carried + response.chat_message.content
        return response

    async def __search_streaming(
        self, request: ChatClientRequest, settings: ToolSearchMethodSettings
    ) -> AsyncIterator[ChatResponse]:
        conversation = request["conversation"]
        found: Set[str] = set()
        # Chunks carry the response's text so far; text a search round already
        # streamed stays in front of the next round's.
        carried = ""
        for search_round in range(MAX_SEARCH_ROUNDS + 1):
            last: Optional[ChatResponse] = None
            previous = carried
            stream = cast(
                AsyncIterator[ChatResponse],
                await self.__client.complete(
                    self.__round_request(
                        request,
                        conversation,
                        settings,
                        found,
                        search_round == MAX_SEARCH_ROUNDS,
                    ),
                    stream=True,
                ),
            )
            async for chunk in stream:
                if carried and chunk.chat_message is not None:
                    chunk.chat_message.content = carried + chunk.chat_message.content
                last = chunk
                # Calls arrive on the stream's final chunk; it is held back until
                # it is known not to be a search.
                if not (chunk.chat_message and chunk.chat_message.tool_call_requests):
                    yield chunk
            if last is None:
                return
            if not self.__search_calls(last):
                if last.chat_message and last.chat_message.tool_call_requests:
                    yield last
                return
            carried = cast(ChatMessage, last.chat_message).content
            searching = cast(ChatMessage, last.chat_message).copy(
                update={"content": carried[len(previous) :]}
            )
            conversation = self.__answer_searches(
                request,
                conversation,
                settings,
                ChatResponse(error=None, chat_message=searching),
                found,
            )
