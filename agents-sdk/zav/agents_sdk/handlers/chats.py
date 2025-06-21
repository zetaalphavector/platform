from typing import Any, Callable, List, Optional, Type

from zav.llm_tracing import TracingBackendFactory
from zav.message_bus import (  # noqa
    CommandHandlerRegistry,
    EventHandlerRegistry,
    Message,
)

from zav.agents_sdk.adapters.event_publishers.event_publisher import (
    AbstractEventPublisher,
)
from zav.agents_sdk.domain.agent_event import AgentEvent
from zav.agents_sdk.domain.agent_registries_factory import AgentRegistriesFactory
from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
from zav.agents_sdk.domain.chat_message import ChatMessage as DomainChatMessage
from zav.agents_sdk.domain.chat_message import (
    ChatMessageSender as DomainChatMessageSender,
)
from zav.agents_sdk.domain.chat_message import (
    FunctionCallRequest as DomainFunctionCallRequest,
)
from zav.agents_sdk.domain.chat_message import (
    FunctionSpec,
)
from zav.agents_sdk.domain.chat_request import ChatRequest
from zav.agents_sdk.handlers import commands, events
from zav.agents_sdk.mem_utils import cleanup_memory
from zav.agents_sdk.security import sanitize_bot_params


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


@CommandHandlerRegistry.register(commands.CreateChatResponse)
async def handle_create(
    cmd: commands.CreateChatResponse,
    queue: List[Message],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
    event_publisher: Optional[AbstractEventPublisher] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
):
    (
        agent_setup_retriever,
        chat_agent_class_registry,
        agent_dependency_registry,
    ) = await agent_registries_factory.create(tenant=cmd.tenant)

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
        },
        agent_dependency_registry=agent_dependency_registry,
        debug_backend=debug_backend,
        publish_event=lambda agent_event: push_event_to_queue(
            cmd, agent_event, event_publisher
        ),
    )
    chat_agent = await chat_agent_factory.create(
        agent_identifier=cmd.chat_request.agent_identifier,
        conversation_context=cmd.chat_request.conversation_context,
        handler_params={
            **sanitize_bot_params(cmd.chat_request.bot_params),
            **({"tenant": cmd.tenant} if cmd.tenant else {}),
            **({"request_headers": cmd.request_headers} if cmd.request_headers else {}),
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
        },
    )

    chat_agent_response = await chat_agent.execute(
        conversation=cmd.chat_request.conversation
    )
    if not chat_agent_response:
        return cmd.chat_request

    result = ChatRequest(
        agent_identifier=cmd.chat_request.agent_identifier,
        conversation=cmd.chat_request.conversation
        + [
            DomainChatMessage(
                sender=DomainChatMessageSender(chat_agent_response.sender),
                content=chat_agent_response.content,
                content_parts=chat_agent_response.content_parts,
                image_uri=chat_agent_response.image_uri,
                evidences=chat_agent_response.evidences,
                function_call_request=(
                    DomainFunctionCallRequest.from_orm(
                        chat_agent_response.function_call_request
                    )
                    if chat_agent_response.function_call_request
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
    del chat_agent
    cleanup_memory()
    return result


@CommandHandlerRegistry.register(commands.CreateChatStream)
async def handle_create_stream(
    cmd: commands.CreateChatStream,
    queue: List[Message],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
    event_publisher: Optional[AbstractEventPublisher] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
):

    (
        agent_setup_retriever,
        chat_agent_class_registry,
        agent_dependency_registry,
    ) = await agent_registries_factory.create(tenant=cmd.tenant)

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
        },
        agent_dependency_registry=agent_dependency_registry,
        debug_backend=debug_backend,
        publish_event=lambda agent_event: push_event_to_queue(
            cmd, agent_event, event_publisher
        ),
    )

    chat_agent = await chat_agent_factory.create_streamable(
        agent_identifier=cmd.chat_request.agent_identifier,
        conversation_context=cmd.chat_request.conversation_context,
        handler_params={
            **sanitize_bot_params(cmd.chat_request.bot_params),
            **({"tenant": cmd.tenant} if cmd.tenant else {}),
            **({"request_headers": cmd.request_headers} if cmd.request_headers else {}),
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
        },
    )

    try:
        chat_agent_response = chat_agent.execute_streaming(
            conversation=cmd.chat_request.conversation
        )
    except NotImplementedError:
        agent_name = (
            chat_agent_factory.agent_setup.agent_name
            if chat_agent_factory.agent_setup
            else cmd.chat_request.agent_identifier
        )
        raise NotImplementedError(
            f"The agent {agent_name} does " "not support streaming yet."
        )

    return chat_agent_response
