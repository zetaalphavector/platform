import asyncio
import json
import os
from datetime import datetime
from hashlib import sha1
from typing import Any, List, Optional

import streamlit as st
from pydantic import BaseModel
from zav.llm_domain import LLMModelType
from zav.message_bus import MessageBus
from zav.object_storage_repo import ObjectRepositoryFactory, ObjectStorageItem

from zav.agents_sdk import (
    AgentSetup,
    AgentSetupRetrieverFromFile,
    ChatMessage,
    ChatMessageSender,
    ContentPartTool,
    ConversationContext,
    CustomContext,
    CustomContextItem,
    DocumentContext,
)
from zav.agents_sdk.adapters import AgentDependencyRegistry
from zav.agents_sdk.adapters.local_agent_registries_factory import (
    LocalAgentRegistriesFactory,
)
from zav.agents_sdk.bootstrap import setup_bootstrap
from zav.agents_sdk.cli.load_chat_agent_factory import (
    from_string as import_chat_agent_class_registry_from_string,
)
from zav.agents_sdk.domain import ChatRequest, RequestHeaders
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry
from zav.agents_sdk.handlers import commands

st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        [data-testid="stToolbar"] {display: none;}
    </style>
""",
    unsafe_allow_html=True,
)

zav_project_dir = os.environ["ZAV_PROJECT_DIR"]
zav_fe_url = os.environ["ZAV_FE_URL"]
zav_agent_setup_src = os.getenv("ZAV_AGENT_SETUP_SRC")
zav_secret_agent_setup_src = os.getenv("ZAV_SECRET_AGENT_SETUP_SRC")
storage_backend = os.environ["STORAGE_BACKEND"]
storage_path = os.environ["STORAGE_PATH"]

object_storage_repo = ObjectRepositoryFactory.create(storage_backend)
import_chat_agent_class_registry_from_string(zav_project_dir)
# No secrets in this retriever, so it's safe to store
safe_agent_setup_retriever = AgentSetupRetrieverFromFile(file_path=zav_agent_setup_src)
agent_setup_retriever = AgentSetupRetrieverFromFile(
    file_path=zav_agent_setup_src, secret_file_path=zav_secret_agent_setup_src
)
# TODO: Think of a better way of passing the api key to RAGElo
os.environ["OPENAI_API_KEY"] = next(
    iter(
        api_key
        for ags in asyncio.run(agent_setup_retriever.list())
        if ags.llm_client_configuration
        and (openai_conf := ags.llm_client_configuration.vendor_configuration.openai)
        and (api_key := openai_conf.openai_api_key.get_unencrypted_secret())
    ),
    "",
)

debug_storage: List[Any] = []
bootstrap = setup_bootstrap(
    agent_registries_factory=LocalAgentRegistriesFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=ChatAgentClassRegistry,
        agent_dependency_registry=AgentDependencyRegistry,
    ),
    debug_backend=debug_storage.append,
)

message_bus = bootstrap.message_bus
TO_CHAT_MESSAGE_NAME = {
    ChatMessageSender.USER: "user",
    ChatMessageSender.BOT: "assistant",
}


class ChatConfigurationItem(BaseModel):
    agent_identifier: str
    agent_setup: AgentSetup
    conversation_context: Optional[ConversationContext] = None

    def hash(self) -> str:
        return str(
            sha1(self.json(exclude_none=True).encode("utf-8")).hexdigest()  # nosec
        )


class ChatMessageItem(BaseModel):
    message: ChatMessage
    debug_storage: Optional[List[Any]] = None


class EvaluatorItem(BaseModel):
    verdict: str
    explanation: str


class ChatEntry(BaseModel):
    chat_configuration_item: Optional[ChatConfigurationItem] = None
    chat_message_item: Optional[ChatMessageItem] = None
    evaluator_item: Optional[EvaluatorItem] = None


class ComputeChatMessageItem(BaseModel):
    message_bus: MessageBus
    chat_configuration_item: ChatConfigurationItem

    class Config:
        arbitrary_types_allowed = True


class TraceFileContent(BaseModel):
    entries: List[ChatEntry] = []


def new_trace_file_name(agent_identifier: str):
    time_now = str(datetime.now().isoformat())
    trace_file_name = (
        f"{storage_path}/agent-traces/{time_now}_{agent_identifier}_trace.json"
    )
    st.session_state.trace_file_name = trace_file_name
    return trace_file_name


def get_trace_file_names():
    object_attributes = asyncio.run(
        object_storage_repo.filter_objects_attributes(
            url_prefix=f"{storage_path}/agent-traces"
        )
    )
    trace_file_names = []
    for object_attribute in object_attributes:
        if object_attribute.url.endswith(".json"):
            trace_file_names.append(object_attribute.url.split("/")[-1])
    return sorted(trace_file_names)


def store_trace_file_content(agent_identifier: str, entries: List[ChatEntry]):
    if "trace_file_name" not in st.session_state:
        trace_file_name = new_trace_file_name(agent_identifier=agent_identifier)
    else:
        trace_file_name = st.session_state.trace_file_name

    asyncio.run(
        object_storage_repo.add(
            ObjectStorageItem(
                url=trace_file_name,
                payload=TraceFileContent(entries=entries)
                .json(indent=2)
                .encode("utf-8"),
            )
        )
    )


def retrieve_trace_file_content(trace_file_name: str):
    url_prefix = f"{storage_path}/agent-traces"
    trace_file_content = asyncio.run(
        object_storage_repo.get(
            f"{url_prefix}/{trace_file_name.replace('agent-traces/', '')}"
            if not trace_file_name.startswith(url_prefix)
            else trace_file_name
        )
    )
    if not trace_file_content:
        raise FileNotFoundError(f"Trace file {trace_file_name} not found")
    return TraceFileContent.parse_raw(trace_file_content.payload)


def start_new_conversation():
    debug_storage.clear()
    st.session_state.entries = []
    if "agent_identifier" in st.session_state:
        new_trace_file_name(agent_identifier=st.session_state.agent_identifier)


def render_tool_content(tool: ContentPartTool):
    if tool.name == "query_creator_agent":
        msg = "Exploring the topic of"
        if tool.params is not None:
            msg += f" {tool.params.get('message', '')}"
        if tool.response is not None:
            msg += f" - {tool.response.get('content', msg)}"
    elif tool.name == "searcher_agent":
        msg = "Searching for "
        if tool.params is not None:
            msg += f"\"{tool.params.get('query', '')}\""
        if tool.response is not None:
            msg += (
                f" Found - {tool.response.get('number_of_relevant_hits_found', 0)}"
                " relevant docs"
            )
    return msg


def render_chat_message_item_content(st_elem, content: ChatMessage):
    parsed_content = ""
    if content.content_parts is not None:
        for content_part in content.content_parts:
            if content_part.type == "tool" and content_part.tool:
                parsed_content = render_tool_content(content_part.tool)
            elif content_part.type == "text" and content_part.text:
                parsed_content = content_part.text
            elif content_part.type == "table":
                parsed_content = content.content
    else:
        parsed_content = content.content
    if content.evidences:
        seen_evidences = set()
        for evidence in content.evidences:
            if evidence.anchor_text and evidence.anchor_text not in seen_evidences:
                seen_evidences.add(evidence.anchor_text)
                doc_id = evidence.document_hit_url.split("property_values=")[1]
                doc_id = doc_id.split("_")[0] + "_0"
                parsed_content = parsed_content.replace(
                    evidence.anchor_text,
                    f"[{evidence.anchor_text}]({zav_fe_url}/documents/{doc_id})",
                )
    return st_elem.markdown(parsed_content)


def render_agent_debug_logs(
    st_elem: Optional[Any],
    debug_storage: Optional[List[Any]] = None,
):
    if st_elem and debug_storage is not None:
        st_elem.caption("Agent debug logs")
        return st_elem.json(debug_storage)
    return None


async def anext(ait):
    return await ait.__anext__()


async def create_chat_response(
    compute_chat_message_item: ComputeChatMessageItem,
    conversation,
    conversation_context,
    chat_message_item_content,
    chat_message_item_debug_panel_status=None,
    **options,
):
    streaming_mode = options.get("streaming_mode", False)

    async def compute_conversation():
        agent_setup = compute_chat_message_item.chat_configuration_item.agent_setup
        agent_configuration = agent_setup.agent_configuration or {}
        results = await compute_chat_message_item.message_bus.handle(
            commands.CreateChatResponse(
                tenant=agent_configuration.get("tenant", ""),
                index_id=agent_configuration.get("index_id", None),
                request_headers=RequestHeaders(),
                chat_request=ChatRequest(
                    agent_identifier=agent_setup.agent_identifier,
                    conversation=conversation,
                    conversation_context=conversation_context,
                ),
            )
        )
        return results.pop(0)

    async def compute_conversation_streaming():
        agent_setup = compute_chat_message_item.chat_configuration_item.agent_setup
        agent_configuration = agent_setup.agent_configuration or {}
        results = await compute_chat_message_item.message_bus.handle(
            commands.CreateChatStream(
                tenant=agent_configuration.get("tenant", ""),
                index_id=agent_configuration.get("index_id", None),
                request_headers=RequestHeaders(),
                chat_request=ChatRequest(
                    agent_identifier=agent_setup.agent_identifier,
                    conversation=conversation,
                    conversation_context=conversation_context,
                ),
            )
        )
        return results.pop(0)

    agent_debug_logs = None
    debug_panel_status = None
    if chat_message_item_debug_panel_status is not None:
        debug_panel_status = chat_message_item_debug_panel_status.status(
            "Debug", expanded=True
        )
        agent_debug_logs = render_agent_debug_logs(debug_panel_status, [])

    if streaming_mode:
        conversation_generator = await compute_conversation_streaming()
        last_message = None
        while True:
            try:
                conversation_task = asyncio.create_task(anext(conversation_generator))
                while not conversation_task.done():
                    if agent_debug_logs and debug_storage:
                        agent_debug_logs.json(debug_storage)

                    await asyncio.sleep(0.01)
                conversation_result = conversation_task.result()
                last_message = conversation_result
                render_chat_message_item_content(
                    chat_message_item_content, last_message
                )
            except StopAsyncIteration:
                break
    else:
        conversation_task = asyncio.create_task(compute_conversation())
        while not conversation_task.done():
            if agent_debug_logs and debug_storage:
                agent_debug_logs.json(debug_storage)

            await asyncio.sleep(0.01)
        conversation_result = conversation_task.result()
        last_message = conversation_result.conversation[-1]
        render_chat_message_item_content(chat_message_item_content, last_message)

    if debug_panel_status:
        debug_panel_status.update(state="complete")
    if last_message:
        return ChatMessageItem(
            message=last_message,
            debug_storage=debug_storage,
        )


def agent_display_name(agent_full_name: str):
    return agent_full_name.split("-")[0] + "-" + agent_full_name.split("-")[1][:4]


def render_chat_configuration_item(
    chat_configuration_item: ChatConfigurationItem,
    render_expander_title: bool = False,
    chat_configuration_key_postfix: str = "",
    **options,
):
    with st.chat_message("Configuration"):
        chat_configuration_item_hash = chat_configuration_item.hash()
        expander_title = (
            agent_display_name(
                chat_configuration_item.agent_identifier
                + "-"
                + chat_configuration_item_hash
            )
            if render_expander_title
            else "Configuration changes"
        )

        with st.expander(expander_title, expanded=False):
            st.subheader("Agent configuration")

            current_eval_chat_config_items = getattr(
                st.session_state, "eval_chat_config_items", {}
            )
            to_eval = st.toggle(
                "Add to evaluator",
                key=chat_configuration_item_hash + chat_configuration_key_postfix,
                value=bool(
                    current_eval_chat_config_items.get(chat_configuration_item_hash)
                ),
            )
            if to_eval is not None:
                if to_eval is True:
                    st.session_state.eval_chat_config_items = {
                        **current_eval_chat_config_items,
                        chat_configuration_item_hash: chat_configuration_item,
                    }
                else:
                    if chat_configuration_item_hash in current_eval_chat_config_items:
                        current_eval_chat_config_items.pop(chat_configuration_item_hash)

            st.caption("Agent")
            st.text(chat_configuration_item.agent_identifier)

            agent_configuration = (
                chat_configuration_item.agent_setup.agent_configuration
            )
            st.caption("Agent configuration")
            st.json(json.dumps(agent_configuration or {}, indent=2))

            st.subheader("LLM model configuration")
            llm_client_configuration = (
                chat_configuration_item.agent_setup.llm_client_configuration
            )
            llm_model_configuration = (
                llm_client_configuration.model_configuration
                if llm_client_configuration
                else None
            )
            st.caption("Name")
            st.text(llm_model_configuration.name if llm_model_configuration else None)
            st.caption("Type")
            st.text(
                llm_model_configuration.type.value if llm_model_configuration else None
            )
            st.caption("Temperature")
            st.text(
                llm_model_configuration.temperature if llm_model_configuration else None
            )
            st.caption("Max tokens")
            st.text(
                llm_model_configuration.max_tokens if llm_model_configuration else None
            )

            st.subheader("Conversation context")
            cc = chat_configuration_item.conversation_context
            if cc and cc.document_context:
                st.caption("Document IDs")
                st.json(
                    json.dumps(
                        (
                            cc.document_context.document_ids
                            if cc and cc.document_context
                            else []
                        ),
                        indent=2,
                    )
                )
                st.caption("Retrieval unit")
                st.text(
                    cc.document_context.retrieval_unit
                    if cc and cc.document_context
                    else None
                )
            elif cc and cc.custom_context:
                st.caption("Custom context items")
                st.json(cc.custom_context.json(indent=2))


def render_chat_message_item(
    chat_message_item: ChatMessageItem,
    **options,
):
    message = chat_message_item.message
    with st.chat_message(TO_CHAT_MESSAGE_NAME[message.sender]):
        render_chat_message_item_content(st, message)
        if message.sender == ChatMessageSender.BOT:
            debug_panel_status = None
            if options.get("print_debug_logs"):
                debug_panel_status = st.status("Debug", expanded=False)
                render_agent_debug_logs(
                    debug_panel_status,
                    chat_message_item.debug_storage,
                )
                debug_panel_status.update(state="complete")
        return chat_message_item


def render_evaluator_item(eval_item: EvaluatorItem):
    with st.chat_message("Evaluator", avatar="⚖️"):
        st.markdown(
            f"Verdict: {eval_item.verdict}\n\nExplanation:\n{eval_item.explanation}"
        )
        return eval_item


def render_compute_chat_message_item(
    previous_chat_messages: List[ChatMessageItem],
    conversation_context: Optional[ConversationContext],
    compute_chat_message_item: ComputeChatMessageItem,
    **options,
):
    with st.chat_message(TO_CHAT_MESSAGE_NAME[ChatMessageSender.BOT]):
        chat_message_item_debug_panel_status = None
        if options.get("print_debug_logs"):
            chat_message_item_debug_panel_status = st.empty()
        chat_message_item_content = st.markdown("Thinking...")
        created_chat_message_item = asyncio.run(
            create_chat_response(
                compute_chat_message_item,
                [chat_message.message for chat_message in previous_chat_messages],
                conversation_context,
                chat_message_item_content,
                chat_message_item_debug_panel_status,
                **options,
            )
        )
        return created_chat_message_item


def render_entry(
    entries: List[ChatEntry],
    entry: Optional[ChatEntry] = None,
    compute_message_item: Optional[ComputeChatMessageItem] = None,
    conversation_context: Optional[ConversationContext] = None,
    **options,
):
    if entry:
        if entry.chat_message_item:
            created_chat_message_item = render_chat_message_item(
                entry.chat_message_item,
                **options,
            )
            entries.append(
                ChatEntry(
                    chat_message_item=created_chat_message_item,
                )
            )
        elif entry.chat_configuration_item:
            render_chat_configuration_item(entry.chat_configuration_item, **options)
            entries.append(entry)
        elif entry.evaluator_item:
            render_evaluator_item(entry.evaluator_item)
            entries.append(entry)
    elif compute_message_item:
        created_chat_message_item = render_compute_chat_message_item(
            [entry.chat_message_item for entry in entries if entry.chat_message_item],
            conversation_context,
            compute_message_item,
            **options,
        )
        entries.append(
            ChatEntry(
                chat_message_item=created_chat_message_item,
            )
        )


st.logo(
    "https://search.zeta-alpha.com/assets/img/zeta-logo-full-white.svg",
    link="https://docs.zeta-alpha.com/gen-ai/customize/getting-started",
    icon_image="https://search.zeta-alpha.com/assets/img/zeta-logo-white.svg",
)

if "trace_file_name" in st.session_state:
    st.caption(f"Saving to trace file: {st.session_state.trace_file_name}")
st.sidebar.page_link("ui_app.py", label="Chat", icon="💬")
st.sidebar.page_link("pages/ui_collect.py", label="Collect", icon="🪣")
st.sidebar.page_link("pages/ui_eval.py", label="RAGElo Evaluation", icon="🔎")
st.sidebar.divider()
with st.sidebar:
    if st.button("New conversation"):
        start_new_conversation()

    sel_existing_trace = st.selectbox(
        "Load previous conversation", get_trace_file_names(), index=None
    )
    if sel_existing_trace:
        trace_file_content = retrieve_trace_file_content(sel_existing_trace)

        chat_configuration_item = next(
            iter(
                entry.chat_configuration_item
                for entry in reversed(trace_file_content.entries)
                if entry.chat_configuration_item
            ),
            None,
        )
        if chat_configuration_item:
            st.session_state.agent_identifier = chat_configuration_item.agent_identifier
            st.session_state.agent_setup = chat_configuration_item.agent_setup
            st.session_state.current_chat_configuration_item_hash = (
                chat_configuration_item.hash()
            )

            safe_agent_setup_retriever.update_agent_setup(
                agent_identifier=st.session_state.agent_identifier,
                agent_setup_patch=chat_configuration_item.agent_setup.dict(
                    exclude_unset=True,
                    exclude_none=True,
                    exclude_defaults=True,
                ),
            )
            agent_setup_retriever.update_agent_setup(
                agent_identifier=st.session_state.agent_identifier,
                agent_setup_patch=chat_configuration_item.agent_setup.dict(
                    exclude_unset=True,
                    exclude_none=True,
                    exclude_defaults=True,
                ),
            )
        start_new_conversation()
        st.session_state.entries = trace_file_content.entries

    with st.expander("Agent configuration", expanded=True):
        all_agent_setups = asyncio.run(agent_setup_retriever.list())
        st.selectbox(
            "Agent",
            set(
                [agent_setup.agent_identifier for agent_setup in all_agent_setups]
                + [agent for agent in ChatAgentClassRegistry.registry]
            ),
            key="agent_identifier",
        )

        # Get agent name with same agent identifier from all_agent_setups
        agent_name = next(
            (
                agent_setup.agent_identifier
                for agent_setup in all_agent_setups
                if agent_setup.agent_identifier == st.session_state.agent_identifier
            )
        )
        agent_setup = asyncio.run(
            safe_agent_setup_retriever.get(
                st.session_state.agent_identifier,
            )
        )
        if agent_setup is None:
            agent_setup = AgentSetup(
                agent_identifier=st.session_state.agent_identifier,
                agent_name=agent_name,
            )
        st.session_state.agent_setup = agent_setup

        agent_configuration = (
            (agent_setup.agent_configuration or {}) if agent_setup else {}
        )

        sel_agent_configuration = st.text_area(
            "Agent configuration", json.dumps(agent_configuration, indent=2)
        )

    with st.expander("LLM model configuration"):
        llm_model_configuration = (
            agent_setup.llm_client_configuration.model_configuration
            if agent_setup and agent_setup.llm_client_configuration
            else None
        )
        sel_name = st.text_input(
            "Name", llm_model_configuration.name if llm_model_configuration else None
        )
        type_options = [e.value for e in LLMModelType]
        sel_type = st.selectbox(
            "Type",
            type_options,
            index=(
                type_options.index(llm_model_configuration.type.value)
                if llm_model_configuration
                else None
            ),
        )
        sel_temperature = st.number_input(
            "Temperature",
            value=(
                llm_model_configuration.temperature if llm_model_configuration else None
            ),
        )
        sel_max_tokens = st.number_input(
            "Max tokens",
            value=(
                llm_model_configuration.max_tokens if llm_model_configuration else None
            ),
        )

    conversation_context: Optional[ConversationContext] = None
    with st.expander("Conversation context"):
        st.caption("Document context")
        sel_document_ids = st.text_area("Document IDs", json.dumps([], indent=2))
        sel_retrieval_unit = st.text_input("Retrieval unit")

        st.caption("Custom context")
        sel_custom_context_items = st.text_area("Items", json.dumps([], indent=2))

    if sel_document_ids and sel_retrieval_unit:
        conversation_context = ConversationContext(
            document_context=DocumentContext(
                document_ids=json.loads(sel_document_ids),
                retrieval_unit=sel_retrieval_unit,
            )
        )
    elif sel_custom_context_items:
        conversation_context = ConversationContext(
            custom_context=CustomContext(
                items=[
                    CustomContextItem(**item)
                    for item in json.loads(sel_custom_context_items)
                ]
            )
        )
    st.session_state.conversation_context = conversation_context

    with st.expander("UI configuration"):
        sel_print_debug_logs = st.toggle("Show debug logs", value=True)
        sel_streaming_mode = st.toggle("Streaming mode", value=False)

    if agent_setup:
        if sel_agent_configuration:
            agent_setup.agent_configuration = json.loads(sel_agent_configuration)
        if agent_setup.llm_client_configuration:
            if sel_name:
                agent_setup.llm_client_configuration.model_configuration.name = sel_name
            if sel_type:
                agent_setup.llm_client_configuration.model_configuration.type = (
                    LLMModelType(sel_type)
                )
            if sel_temperature:
                agent_setup.llm_client_configuration.model_configuration.temperature = (
                    sel_temperature
                )
            if sel_max_tokens:
                agent_setup.llm_client_configuration.model_configuration.max_tokens = (
                    int(sel_max_tokens)
                )
        st.session_state.agent_setup = agent_setup

        safe_agent_setup_retriever.update_agent_setup(
            agent_identifier=agent_setup.agent_identifier,
            agent_setup_patch=agent_setup.dict(
                exclude_unset=True,
                exclude_none=True,
                exclude_defaults=True,
            ),
        )
        agent_setup_retriever.update_agent_setup(
            agent_identifier=agent_setup.agent_identifier,
            agent_setup_patch=agent_setup.dict(
                exclude_unset=True,
                exclude_none=True,
                exclude_defaults=True,
            ),
        )


if "agent_identifier" not in st.session_state or "agent_setup" not in st.session_state:
    st.toast("Please select an agent")
else:
    if "entries" not in st.session_state:
        start_new_conversation()

    for entry in st.session_state.entries:
        render_entry(
            entries=[],
            entry=entry,
            print_debug_logs=sel_print_debug_logs,
            streaming_mode=sel_streaming_mode,
        )

    if content := st.chat_input("What is up?"):
        chat_configuration_item = ChatConfigurationItem(
            agent_identifier=st.session_state.agent_identifier,
            agent_setup=st.session_state.agent_setup,
            conversation_context=st.session_state.conversation_context,
        )
        chat_configuration_item_hash = chat_configuration_item.hash()
        if (
            "current_chat_configuration_item_hash" not in st.session_state
            or st.session_state.current_chat_configuration_item_hash
            != chat_configuration_item_hash
        ):
            st.session_state.current_chat_configuration_item_hash = (
                chat_configuration_item_hash
            )
            render_entry(
                entries=st.session_state.entries,
                entry=ChatEntry(chat_configuration_item=chat_configuration_item),
                print_debug_logs=sel_print_debug_logs,
                streaming_mode=sel_streaming_mode,
            )
        render_entry(
            entries=st.session_state.entries,
            entry=ChatEntry(
                chat_message_item=ChatMessageItem(
                    message=ChatMessage(
                        sender=ChatMessageSender.USER,
                        content=content,
                    )
                )
            ),
            print_debug_logs=sel_print_debug_logs,
            streaming_mode=sel_streaming_mode,
        )

        render_entry(
            entries=st.session_state.entries,
            compute_message_item=ComputeChatMessageItem(
                message_bus=message_bus,
                chat_configuration_item=chat_configuration_item,
            ),
            conversation_context=conversation_context,
            print_debug_logs=sel_print_debug_logs,
            streaming_mode=sel_streaming_mode,
        )
        store_trace_file_content(
            agent_identifier=st.session_state.agent_identifier,
            entries=st.session_state.entries,
        )
