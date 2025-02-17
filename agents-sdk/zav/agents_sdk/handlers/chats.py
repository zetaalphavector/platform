from typing import Any, Callable, List, Optional, Type

from zav.llm_tracing import Span, Trace, TracingBackendFactory
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
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetup
from zav.agents_sdk.domain.chat_agent_factory import ChatAgentFactory
from zav.agents_sdk.domain.chat_message import ChatMessage as DomainChatMessage
from zav.agents_sdk.domain.chat_message import (
    ChatMessageSender as DomainChatMessageSender,
)
from zav.agents_sdk.domain.chat_message import (
    FunctionCallRequest as DomainFunctionCallRequest,
)
from zav.agents_sdk.domain.chat_message import FunctionSpec
from zav.agents_sdk.domain.chat_request import ChatRequest
from zav.agents_sdk.domain.request_headers import RequestHeaders
from zav.agents_sdk.handlers import commands, events


def init_span(
    tracing_backend_factory: Type[TracingBackendFactory],
    agent_setup: AgentSetup,
    chat_request: ChatRequest,
    request_headers: RequestHeaders,
    tenant: str,
    index_id: Optional[str] = None,
) -> Optional[Span]:
    span: Optional[Span] = None
    if tracing_config := agent_setup.tracing_configuration:
        tracing_backend = tracing_backend_factory.create(config=tracing_config)
        span = Trace(tracing_backend=tracing_backend).new(
            name="agent-response",
            attributes={
                "metadata": {"agent_identifier": chat_request.agent_identifier},
            },
            trace_state={
                "tenant": tenant,
                **({"index_id": index_id} if index_id else {}),
                **(
                    {"user_id": request_headers.requester_uuid}
                    if request_headers.requester_uuid
                    else {}
                ),
            },
        )

    return span


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
    agent_setup_retriever, chat_agent_class_registry, agent_dependency_registry = (
        await agent_registries_factory.create(tenant=cmd.tenant)
    )
    agent_setup = await agent_setup_retriever.get(
        agent_identifier=cmd.chat_request.agent_identifier
    )
    if not agent_setup:
        raise ValueError(f"Unknown agent: {cmd.chat_request.agent_identifier}")

    span = init_span(
        tracing_backend_factory=tracing_backend_factory,
        agent_setup=agent_setup,
        chat_request=cmd.chat_request,
        request_headers=cmd.request_headers,
        tenant=cmd.tenant,
        index_id=cmd.index_id,
    )

    chat_agent = await ChatAgentFactory.create(
        agent_name=agent_setup.agent_name,
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=chat_agent_class_registry,
        agent_dependency_registry=agent_dependency_registry,
        debug_backend=debug_backend,
        agent_setup=agent_setup,
        handler_params={
            **({"tenant": cmd.tenant} if cmd.tenant else {}),
            **({"request_headers": cmd.request_headers} if cmd.request_headers else {}),
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
            **(cmd.chat_request.bot_params if cmd.chat_request.bot_params else {}),
        },
        conversation_context=cmd.chat_request.conversation_context,
        span=span,
        publish_event=lambda agent_event: push_event_to_queue(
            cmd, agent_event, event_publisher
        ),
    )

    chat_agent_response = await chat_agent.execute(
        conversation=cmd.chat_request.conversation
    )
    if not chat_agent_response:
        return cmd.chat_request

    return ChatRequest(
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
                    if agent_setup
                    and agent_setup.agent_configuration
                    and (
                        function_specs := agent_setup.agent_configuration.get(
                            "function_specs", None
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


@CommandHandlerRegistry.register(commands.CreateChatStream)
async def handle_create_stream(
    cmd: commands.CreateChatStream,
    queue: List[Message],
    agent_registries_factory: AgentRegistriesFactory,
    tracing_backend_factory: Type[TracingBackendFactory],
    event_publisher: Optional[AbstractEventPublisher] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
):
    agent_setup_retriever, chat_agent_class_registry, agent_dependency_registry = (
        await agent_registries_factory.create(tenant=cmd.tenant)
    )
    agent_setup = await agent_setup_retriever.get(
        agent_identifier=cmd.chat_request.agent_identifier
    )
    if not agent_setup:
        raise ValueError(f"Unknown agent: {cmd.chat_request.agent_identifier}")

    span = init_span(
        tracing_backend_factory=tracing_backend_factory,
        agent_setup=agent_setup,
        chat_request=cmd.chat_request,
        request_headers=cmd.request_headers,
        tenant=cmd.tenant,
        index_id=cmd.index_id,
    )

    chat_agent = await ChatAgentFactory.create_streamable(
        agent_name=agent_setup.agent_name,
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=chat_agent_class_registry,
        agent_dependency_registry=agent_dependency_registry,
        debug_backend=debug_backend,
        agent_setup=agent_setup,
        handler_params={
            **({"tenant": cmd.tenant} if cmd.tenant else {}),
            **({"request_headers": cmd.request_headers} if cmd.request_headers else {}),
            **({"index_id": cmd.index_id} if cmd.index_id else {}),
            **(cmd.chat_request.bot_params if cmd.chat_request.bot_params else {}),
        },
        conversation_context=cmd.chat_request.conversation_context,
        span=span,
        publish_event=lambda agent_event: push_event_to_queue(
            cmd, agent_event, event_publisher
        ),
    )

    try:
        chat_agent_response = chat_agent.execute_streaming(
            conversation=cmd.chat_request.conversation
        )
    except NotImplementedError:
        raise NotImplementedError(
            f"The agent {agent_setup.agent_name} does not support streaming yet."
        )

    return chat_agent_response
