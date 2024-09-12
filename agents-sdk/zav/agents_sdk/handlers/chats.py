from typing import Any, Callable, List, Optional, Type

from zav.message_bus import (  # noqa
    CommandHandlerRegistry,
    EventHandlerRegistry,
    Message,
)

from zav.agents_sdk.domain.agent_dependency import AgentDependencyRegistry
from zav.agents_sdk.domain.agent_setup_retriever import AgentSetupRetriever
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
from zav.agents_sdk.handlers import commands


@CommandHandlerRegistry.register(commands.CreateChatResponse)
async def handle_create(
    cmd: commands.CreateChatResponse,
    queue: List[Message],
    agent_setup_retriever: AgentSetupRetriever,
    chat_agent_factory: Type[ChatAgentFactory],
    agent_dependency_registry: Optional[Type[AgentDependencyRegistry]] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
):
    agent_setup = await agent_setup_retriever.get(
        tenant=cmd.tenant, agent_identifier=cmd.chat_request.agent_identifier
    )
    if not agent_setup:
        raise ValueError(f"Unknown agent: {cmd.chat_request.agent_identifier}")

    chat_agent = await chat_agent_factory.create(
        agent_name=agent_setup.agent_name,
        agent_setup_retriever=agent_setup_retriever,
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
    agent_setup_retriever: AgentSetupRetriever,
    chat_agent_factory: Type[ChatAgentFactory],
    agent_dependency_registry: Optional[Type[AgentDependencyRegistry]] = None,
    debug_backend: Optional[Callable[[Any], Any]] = None,
):
    agent_setup = await agent_setup_retriever.get(
        tenant=cmd.tenant, agent_identifier=cmd.chat_request.agent_identifier
    )
    if not agent_setup:
        raise ValueError(f"Unknown agent: {cmd.chat_request.agent_identifier}")

    chat_agent = await chat_agent_factory.create_streamable(
        agent_name=agent_setup.agent_name,
        agent_setup_retriever=agent_setup_retriever,
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
