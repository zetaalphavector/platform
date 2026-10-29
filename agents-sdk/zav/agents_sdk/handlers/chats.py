import asyncio
from typing import Any, AsyncGenerator, Awaitable, Callable, List, Optional, Type

from zav.llm_tracing import TracingBackendFactory
from zav.logging import logger
from zav.message_bus import (  # noqa
    CommandHandlerRegistry,
    EventHandlerRegistry,
    HandlerException,
    Message,
    StreamCommandHandlerRegistry,
)

from zav.agents_sdk.adapters.event_publishers.event_publisher import (
    AbstractEventPublisher,
)
from zav.agents_sdk.domain.agent_async_worker import AgentAsyncWorker
from zav.agents_sdk.domain.agent_event import AgentEvent
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.domain.agent_setup_retriever import require_agent_setup
from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
from zav.agents_sdk.domain.chat_agent_state_store import (
    ChatAgentStateStore,
    DependencyState,
)
from zav.agents_sdk.domain.chat_conversation_store import ChatConversationStore
from zav.agents_sdk.domain.chat_message import ChatMessage as DomainChatMessage
from zav.agents_sdk.domain.chat_message import (
    ChatMessageSender as DomainChatMessageSender,
)
from zav.agents_sdk.domain.chat_message import (
    FunctionCallRequest as DomainFunctionCallRequest,
)
from zav.agents_sdk.domain.chat_message import (
    FunctionCallResponse as DomainFunctionCallResponse,
)
from zav.agents_sdk.domain.chat_message import (
    FunctionSpec,
)
from zav.agents_sdk.domain.chat_request import ChatRequest
from zav.agents_sdk.domain.message_inbox import InboxItem, MessageInboxStore
from zav.agents_sdk.domain.turn_inbox import TurnInbox
from zav.agents_sdk.handlers import commands, events
from zav.agents_sdk.mem_utils import cleanup_memory
from zav.agents_sdk.security import sanitize_bot_params


class ChatResponseTimeoutError(Exception):
    """A non-streaming chat turn exceeded its caller-supplied `request_timeout`.

    The in-flight turn is cancelled when this is raised. Distinct from a server
    fault so a caller (e.g. the pipeline agent processor) can treat it as
    transient rather than a hard failure.
    """


class ChatMessageAlreadyReadError(HandlerException):
    """The message left the conversation's inbox: a turn read it."""


async def push_event_to_queue(
    cmd: commands.CreateChatResponse,
    agent_event: AgentEvent,
    event_publisher: Optional[AbstractEventPublisher] = None,
):
    if event_publisher is None:
        return

    event = events.CreatedAgentRequest(
        tenant=cmd.tenant,
        index_id=cmd.index_id,
        request_headers=cmd.request_headers.dict(),
        agent_identifier=agent_event.recipient_agent_identifier,
        **agent_event.payload,
    )
    await event_publisher.publish_event(event)


def make_dependency_state_saver(
    cmd: commands.CreateChatResponse,
    chat_agent_factory: ChatAgentFactory,
    agent_state_store: ChatAgentStateStore,
) -> Callable[[], Awaitable[None]]:
    last_saved_state: Optional[DependencyState] = None

    async def save_dependency_state() -> None:
        nonlocal last_saved_state
        state = await chat_agent_factory.dump_dependency_state()
        # Each resumable dependency returns the same object while its state is
        # unchanged (see the dump cache in ``ResumableZAVChatCompletionClient``),
        # so a per-key identity check detects "nothing new" in O(keys) without
        # deep-walking the whole blob.
        unchanged = (
            last_saved_state is not None
            and state.keys() == last_saved_state.keys()
            and all(state[key] is last_saved_state[key] for key in state)
        )
        if unchanged:
            return
        await agent_state_store.save(
            tenant=cmd.tenant,
            user_uuid=cmd.request_headers.requester_uuid,
            session_id=cmd.session_id,
            state=state,
        )
        last_saved_state = state

    return save_dependency_state


def _require_agent_state_store(
    agent_state_store: Optional[ChatAgentStateStore],
) -> ChatAgentStateStore:
    # Stateful turns load/save the agent's working state, so the store is
    # mandatory for them; a stateless turn never reaches here.
    if agent_state_store is None:
        raise ValueError("agent_state_store must be provided for stateful operations")
    return agent_state_store


async def resolve_turn_conversation(
    cmd: commands.CreateChatResponse,
    dependency_state: Optional[DependencyState],
    chat_agent_factory: ChatAgentFactory,
    chat_conversation_store: Optional[ChatConversationStore],
    new_messages: List[DomainChatMessage],
) -> List[DomainChatMessage]:
    # On resume the persisted transcript already holds the user's turn, so the
    # agent runs with only what arrived since and the model/tool loop re-drives
    # from the last checkpoint. With no checkpoint yet (e.g. a crash before the
    # first save) the stored transcript, which holds the user's turn, starts it
    # again below.
    if cmd.resume and dependency_state:
        return new_messages
    # Two situations need the full stored transcript rather than the single new
    # turn the client sent:
    #
    #   * a non-resumable agent never replays a transcript of its own, so every
    #     stateful follow-up must run over the stored history — the client keeps
    #     requests small by sending only the new turn (record_user_turn appended
    #     it above), so without this the agent would forget every earlier turn;
    #     and
    #   * a transcript-resuming agent on its first turn against a conversation that
    #     already has history but no checkpoint yet. A legacy note adopted in place
    #     (see SQLChatConversationStore) seeds the store with the prior transcript
    #     while the agent state starts empty, so there is nothing to restore from.
    #     Feeding the stored history this once captures it into the initial
    #     checkpoint; once a checkpoint exists the agent restores its own
    #     transcript and returns to the new-turn-only path below.
    #
    # On a fresh native conversation the store holds exactly the new turn, so this
    # returns the same single message the client sent — no behavioral change.
    if (
        cmd.stateful
        and chat_conversation_store is not None
        and (not chat_agent_factory.resumes_conversation or not dependency_state)
    ):
        stored = await chat_conversation_store.get_transcript(
            tenant=cmd.tenant,
            requester_uuid=cmd.request_headers.requester_uuid,
            session_id=cmd.session_id,
        )
        if stored:
            return stored
    # Nothing stored the user's turn (no checkpoint, no conversation store), so
    # a resume runs it again from the request.
    if cmd.resume and not new_messages:
        return cmd.chat_request.conversation
    return new_messages


async def record_user_turn(
    cmd: commands.CreateChatResponse,
    chat_conversation_store: Optional[ChatConversationStore],
    new_messages: List[DomainChatMessage],
) -> None:
    """Open the conversation (creating it on the first turn / migration) and
    append the turn's new user message(s) before the agent runs, so the user's
    turn is durable even if the producer crashes mid-answer."""
    if not cmd.stateful or chat_conversation_store is None or not new_messages:
        return
    await chat_conversation_store.append_messages(
        tenant=cmd.tenant,
        requester_uuid=cmd.request_headers.requester_uuid,
        session_id=cmd.session_id,
        agent_identifier=cmd.chat_request.agent_identifier,
        messages=new_messages,
        user_roles=cmd.request_headers.user_roles,
        user_tenants=cmd.request_headers.user_tenants,
        user_agent_id=cmd.chat_request.user_agent_id,
        bot_params=cmd.chat_request.bot_params,
        last_turn_status="running",
        conversation_context=cmd.chat_request.conversation_context,
    )


def request_messages(cmd: commands.CreateChatResponse) -> List[DomainChatMessage]:
    """The messages a request brings to its turn. On resume the user turn is
    already stored and the body is ignored."""
    return [] if cmd.resume else cmd.chat_request.conversation


async def take_turn_messages(
    cmd: commands.CreateChatStream,
    message_inbox_store: MessageInboxStore,
    turn_inbox: TurnInbox,
) -> List[DomainChatMessage]:
    """The request's messages, and what a send brought, go through the
    conversation's inbox like any other, so the turn starts with everything
    nobody has read yet: messages a previous turn left unread included."""
    items = [InboxItem(message=message) for message in request_messages(cmd)]
    for item in items + cmd.inbox_items:
        await message_inbox_store.append(
            tenant=cmd.tenant,
            requester_uuid=cmd.request_headers.requester_uuid,
            session_id=cmd.session_id,
            item=item,
        )
    return await turn_inbox.messages()


async def record_bot_turn(
    cmd: commands.CreateChatResponse,
    chat_conversation_store: Optional[ChatConversationStore],
    queue: List[Message],
    message: DomainChatMessage,
) -> None:
    """Append the completed assistant message to the conversation and enqueue a
    ``ConversationUpdated`` event. The event is queued, not published inline: the
    message bus dispatches it to the registered in-process handler, which pushes
    it outward, and the consumer mirrors the transcript into the saved-results
    note off the request path. Queuing keeps this on the standard event path
    instead of reaching for the publisher from inside the command handler."""
    if not cmd.stateful or chat_conversation_store is None:
        return
    await chat_conversation_store.append_messages(
        tenant=cmd.tenant,
        requester_uuid=cmd.request_headers.requester_uuid,
        session_id=cmd.session_id,
        agent_identifier=cmd.chat_request.agent_identifier,
        messages=[message],
        user_roles=cmd.request_headers.user_roles,
        user_tenants=cmd.request_headers.user_tenants,
        user_agent_id=cmd.chat_request.user_agent_id,
        bot_params=cmd.chat_request.bot_params,
        last_turn_status="complete",
        # The bot turn appends to a conversation the user turn already opened;
        # if a concurrent DELETE removed it mid-stream, do not recreate a ghost.
        create_if_absent=False,
    )
    queue.append(
        events.ConversationUpdated(
            tenant=cmd.tenant,
            # The note mirror is not index-scoped (the conversation is keyed by
            # session); the saved-results note is created/updated without an
            # index. ``index_id`` stays on EventBase only for the agent/batch
            # events that scope retrieval.
            index_id=None,
            request_headers=cmd.request_headers.dict(),
            session_id=cmd.session_id,
            agent_identifier=cmd.chat_request.agent_identifier,
        )
    )


async def record_turn_errored(
    cmd: commands.CreateChatResponse,
    chat_conversation_store: Optional[ChatConversationStore],
) -> None:
    """Move the conversation's ``last_turn_status`` to ``errored`` when the agent
    raises mid-stream, so the conversation list does not show a permanently
    ``running`` turn. The partial answer is intentionally not persisted (a single
    completer fences the answer); only the status changes. Best-effort — a
    failure here must never mask the original error."""
    if not cmd.stateful or chat_conversation_store is None:
        return
    try:
        await chat_conversation_store.set_last_turn_status(
            tenant=cmd.tenant,
            requester_uuid=cmd.request_headers.requester_uuid,
            session_id=cmd.session_id,
            last_turn_status="errored",
        )
    except Exception:
        logger.exception("Failed to mark conversation %s errored", cmd.session_id)


@CommandHandlerRegistry.register(commands.CreateChatResponse)
async def handle_create(
    cmd: commands.CreateChatResponse,
    queue: List[Message],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
    agent_state_store: Optional[ChatAgentStateStore] = None,
    chat_conversation_store: Optional[ChatConversationStore] = None,
    event_publisher: Optional[AbstractEventPublisher] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
):
    (
        agent_setup_retriever,
        chat_agent_class_registry,
        agent_dependency_registry,
        llm_configuration_store,
        secret_store,
    ) = await agent_registries_factory.create(
        tenant=cmd.tenant, request_headers=cmd.request_headers
    )
    # Before anything of the turn is stored: a caller refused the agent leaves
    # no message behind.
    await require_agent_setup(agent_setup_retriever, cmd.chat_request.agent_identifier)
    dependency_state = None
    if cmd.stateful:
        store = _require_agent_state_store(agent_state_store)
        dependency_state = await store.load(
            tenant=cmd.tenant,
            user_uuid=cmd.request_headers.requester_uuid,
            session_id=cmd.session_id,
        )

    new_messages = request_messages(cmd)
    await record_user_turn(cmd, chat_conversation_store, new_messages)

    chat_agent_factory = ChatAgentFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=chat_agent_class_registry,
        tracing_backend_factory=tracing_backend_factory,
        trace_state_params={
            "tenant": cmd.tenant,
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
            **(
                {"user_id": cmd.request_headers.requester_uuid}
                if cmd.request_headers.requester_uuid
                else {}
            ),
            "session_id": cmd.session_id,
        },
        agent_dependency_registry=agent_dependency_registry,
        llm_configuration_store=llm_configuration_store,
        secret_store=secret_store,
        debug_backend=debug_backend,
        publish_event=lambda agent_event: push_event_to_queue(
            cmd, agent_event, event_publisher
        ),
        message_id=cmd.message_id,
        session_id=cmd.session_id,
        dependency_state=dependency_state,
    )
    chat_agent = await chat_agent_factory.create(
        agent_identifier=cmd.chat_request.agent_identifier,
        conversation_context=cmd.chat_request.conversation_context,
        handler_params={
            **sanitize_bot_params(cmd.chat_request.bot_params),
            **({"tenant": cmd.tenant} if cmd.tenant else {}),
            **({"request_headers": cmd.request_headers}),
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
            **({"session_id": cmd.session_id} if cmd.session_id else {}),
        },
    )

    state_worker: Optional[AgentAsyncWorker] = None
    if cmd.stateful:
        store = _require_agent_state_store(agent_state_store)
        state_worker = AgentAsyncWorker(
            make_dependency_state_saver(cmd, chat_agent_factory, store)
        )

    try:
        conversation = await resolve_turn_conversation(
            cmd,
            dependency_state,
            chat_agent_factory,
            chat_conversation_store,
            new_messages,
        )
        if cmd.request_timeout is not None:
            # Bound the turn. wait_for cancels execute() on expiry, which unwinds
            # the in-flight async model call and closes its connection.
            chat_agent_response = await asyncio.wait_for(
                chat_agent.execute(conversation=conversation),
                timeout=cmd.request_timeout,
            )
        else:
            chat_agent_response = await chat_agent.execute(conversation=conversation)
    except asyncio.TimeoutError as e:
        # The turn exceeded its caller-supplied budget. Move the conversation off
        # `running`, then surface a typed timeout the caller can distinguish from
        # a server fault. Without a budget the timeout is a dependency's own and
        # stays a server fault.
        await record_turn_errored(cmd, chat_conversation_store)
        if cmd.request_timeout is None:
            raise
        raise ChatResponseTimeoutError(
            f"Chat response exceeded its {cmd.request_timeout}s budget."
        ) from e
    except Exception:
        # Mirror handle_create_stream: an agent failure must move the
        # conversation off `running` so the list view does not show a
        # permanently in-flight turn. Best-effort; re-raise the original error.
        await record_turn_errored(cmd, chat_conversation_store)
        raise
    finally:
        if state_worker is not None:
            await state_worker.finish()

    if not chat_agent_response:
        return cmd.chat_request

    result = ChatRequest(
        agent_identifier=cmd.chat_request.agent_identifier,
        conversation=cmd.chat_request.conversation
        + [
            DomainChatMessage(
                sender=DomainChatMessageSender(chat_agent_response.sender),
                content=chat_agent_response.content,
                message_id=chat_agent_response.message_id,
                session_id=chat_agent_response.session_id,
                content_parts=chat_agent_response.content_parts,
                image_uri=chat_agent_response.image_uri,
                evidences=chat_agent_response.evidences,
                reasoning_summary=chat_agent_response.reasoning_summary,
                llm_configuration_name=chat_agent_factory.llm_configuration_name,
                function_call_request=(
                    DomainFunctionCallRequest.from_orm(
                        chat_agent_response.function_call_request
                    )
                    if chat_agent_response.function_call_request
                    else None
                ),
                function_call_response=(
                    DomainFunctionCallResponse.from_orm(
                        chat_agent_response.function_call_response
                    )
                    if chat_agent_response.function_call_response
                    else None
                ),
                function_specs=(
                    FunctionSpec(**function_specs)
                    if chat_agent_factory.agent_setup
                    and chat_agent_factory.agent_setup.agent_configuration
                    and (
                        function_specs := (
                            chat_agent_factory.agent_setup.agent_configuration.get(
                                "function_specs", None
                            )
                        )
                    )
                    and chat_agent_response.function_call_request
                    else None
                ),
            )
        ],
        conversation_context=cmd.chat_request.conversation_context,
        bot_params=cmd.chat_request.bot_params,
    )
    await record_bot_turn(cmd, chat_conversation_store, queue, result.conversation[-1])
    del chat_agent
    cleanup_memory()
    return result


@StreamCommandHandlerRegistry.register(commands.CreateChatStream)
async def handle_create_stream(
    cmd: commands.CreateChatStream,
    queue: List[Message],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
    agent_state_store: Optional[ChatAgentStateStore] = None,
    chat_conversation_store: Optional[ChatConversationStore] = None,
    event_publisher: Optional[AbstractEventPublisher] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
    message_inbox_store: Optional[MessageInboxStore] = None,
) -> AsyncGenerator[Any, None]:
    (
        agent_setup_retriever,
        chat_agent_class_registry,
        agent_dependency_registry,
        llm_configuration_store,
        secret_store,
    ) = await agent_registries_factory.create(
        tenant=cmd.tenant, request_headers=cmd.request_headers
    )
    # Before anything of the turn is stored: a caller refused the agent leaves
    # no message behind.
    await require_agent_setup(agent_setup_retriever, cmd.chat_request.agent_identifier)
    dependency_state = None
    if cmd.stateful:
        store = _require_agent_state_store(agent_state_store)
        dependency_state = await store.load(
            tenant=cmd.tenant,
            user_uuid=cmd.request_headers.requester_uuid,
            session_id=cmd.session_id,
        )

    # An interactive turn starts with every message in the conversation's
    # inbox: a send to an idle conversation leaves its message only there.
    # Approval decisions arrive there too. Whether the turn also reads the
    # messages sent while it runs is the agent's queued messages provider's
    # call; unread ones stay for the next turn.
    turn_inbox: Optional[TurnInbox] = None
    new_messages = request_messages(cmd)
    if cmd.stateful and message_inbox_store is not None:
        turn_inbox = TurnInbox(
            message_inbox_store,
            tenant=cmd.tenant,
            requester_uuid=cmd.request_headers.requester_uuid,
            session_id=cmd.session_id,
        )
        new_messages = await take_turn_messages(cmd, message_inbox_store, turn_inbox)
        if not new_messages and not cmd.resume:
            return

    await record_user_turn(cmd, chat_conversation_store, new_messages)

    chat_agent_factory = ChatAgentFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=chat_agent_class_registry,
        tracing_backend_factory=tracing_backend_factory,
        trace_state_params={
            "tenant": cmd.tenant,
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
            **(
                {"user_id": cmd.request_headers.requester_uuid}
                if cmd.request_headers.requester_uuid
                else {}
            ),
            "session_id": cmd.session_id,
        },
        agent_dependency_registry=agent_dependency_registry,
        llm_configuration_store=llm_configuration_store,
        secret_store=secret_store,
        debug_backend=debug_backend,
        publish_event=lambda agent_event: push_event_to_queue(
            cmd, agent_event, event_publisher
        ),
        message_id=cmd.message_id,
        session_id=cmd.session_id,
        dependency_state=dependency_state,
        turn_inbox=turn_inbox,
    )

    chat_agent = await chat_agent_factory.create_streamable(
        agent_identifier=cmd.chat_request.agent_identifier,
        conversation_context=cmd.chat_request.conversation_context,
        handler_params={
            **sanitize_bot_params(cmd.chat_request.bot_params),
            **({"tenant": cmd.tenant} if cmd.tenant else {}),
            **({"request_headers": cmd.request_headers}),
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
            **({"session_id": cmd.session_id} if cmd.session_id else {}),
        },
    )

    turn_conversation = await resolve_turn_conversation(
        cmd,
        dependency_state,
        chat_agent_factory,
        chat_conversation_store,
        new_messages,
    )
    try:
        agent_generator = chat_agent.execute_streaming(conversation=turn_conversation)
    except NotImplementedError:
        agent_name = (
            chat_agent_factory.agent_setup.agent_name
            if chat_agent_factory.agent_setup
            else cmd.chat_request.agent_identifier
        )
        raise NotImplementedError(
            f"The agent {agent_name} does " "not support streaming yet."
        )

    state_worker: Optional[AgentAsyncWorker] = None
    if cmd.stateful:
        store = _require_agent_state_store(agent_state_store)
        state_worker = AgentAsyncWorker(
            make_dependency_state_saver(cmd, chat_agent_factory, store)
        )

    llm_configuration_name = chat_agent_factory.llm_configuration_name
    last_message: Optional[DomainChatMessage] = None
    try:
        async for message in agent_generator:
            if message.sender == DomainChatMessageSender.BOT:
                message.llm_configuration_name = llm_configuration_name
            last_message = message
            yield message
        # The stream drained to completion: copy the final cumulative assistant
        # message into the conversation and fire the update event. Left out of
        # ``finally`` on purpose — a cancelled/errored turn must not persist a
        # partial answer (the supervision lease fences a single completer).
        if last_message is not None:
            await record_bot_turn(cmd, chat_conversation_store, queue, last_message)
    except Exception:
        # The agent raised mid-stream: move the conversation off ``running`` so
        # the list view does not show a permanently in-flight turn, then re-raise
        # so the producer seals ``errored`` and the SSE error frame fires. Cancel
        # and run-budget arrive as ``GeneratorExit`` (not ``Exception``) and pass
        # through to the producer's seal instead.
        await record_turn_errored(cmd, chat_conversation_store)
        raise
    finally:
        if state_worker is not None:
            await state_worker.finish()
        if turn_inbox is not None:
            await turn_inbox.return_unread()


@CommandHandlerRegistry.register(commands.RemoveChatMessage)
async def handle_remove_message(
    cmd: commands.RemoveChatMessage,
    queue: List[Message],
    message_inbox_store: Optional[MessageInboxStore] = None,
) -> None:
    if message_inbox_store is None:
        raise ValueError(
            "Removing a sent message needs message_inbox_store; this server runs "
            "without it (stateless)"
        )
    removed = await message_inbox_store.remove(
        tenant=cmd.tenant,
        requester_uuid=cmd.request_headers.requester_uuid,
        session_id=cmd.session_id,
        message_id=cmd.message_id,
    )
    if not removed:
        raise ChatMessageAlreadyReadError(
            f"Message {cmd.message_id} is no longer in the conversation's inbox"
        )
