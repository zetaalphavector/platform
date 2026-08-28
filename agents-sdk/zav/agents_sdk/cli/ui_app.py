import asyncio
import base64
import html
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import streamlit as st
from zav.llm_domain import LLMClientConfiguration, LLMModelType, LLMProviderName
from zav.object_storage_repo import ObjectRepositoryFactory, ObjectStorageItem
from zav.pydantic_compat import PYDANTIC_V2

import zav.agents_sdk.agents  # noqa: F401 - registers built-in Agent class
from zav.agents_sdk import (
    AgentSetup,
    AgentSetupRetrieverFromFile,
    ChatMessage,
    ChatMessageSender,
    ContentPart,
    ContentPartTool,
    ConversationContext,
    CustomContext,
    CustomContextItem,
    DocumentContext,
    TagContext,
)
from zav.agents_sdk.adapters import AgentDependencyRegistry
from zav.agents_sdk.adapters.agent_state import LocalFileChatAgentStateStore
from zav.agents_sdk.adapters.local_agent_registries_factory import (
    LocalAgentRegistriesFactory,
)
from zav.agents_sdk.bootstrap import setup_bootstrap
from zav.agents_sdk.cli.load_chat_agent_factory import (
    from_string as import_chat_agent_class_registry_from_string,
)
from zav.agents_sdk.cli.local_mcp_oauth import setup_local_mcp_oauth
from zav.agents_sdk.cli.models import (
    ChatConfigurationItem,
    ChatEntry,
    ChatMessageItem,
    ComputeChatMessageItem,
    EvaluatorItem,
    TraceFileContent,
)
from zav.agents_sdk.domain import ChatRequest, RequestHeaders
from zav.agents_sdk.domain.chat_agent_registry import ChatAgentClassRegistry
from zav.agents_sdk.domain.chat_message import (
    extract_internal_document_id_from_evidence_url,
)
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
local_tenant = os.getenv("ZAV_TENANT", "zetaalpha")
local_requester_uuid = os.getenv("ZAV_REQUESTER_UUID", "local-user")
MCP_OAUTH_CALLBACK_PATH = "/mcp/oauth/callback"

object_storage_repo = ObjectRepositoryFactory.create(storage_backend)
agent_state_store = LocalFileChatAgentStateStore(
    base_path=os.path.join(storage_path, "agent-traces", "state")
)

if os.path.isfile(os.path.join(zav_project_dir, "__init__.py")):
    import_chat_agent_class_registry_from_string(zav_project_dir)
# No secrets in this retriever, so it's safe to store
safe_agent_setup_retriever = AgentSetupRetrieverFromFile(file_path=zav_agent_setup_src)
agent_setup_retriever = AgentSetupRetrieverFromFile(
    file_path=zav_agent_setup_src, secret_file_path=zav_secret_agent_setup_src
)


def llm_configuration_for(
    agent_setup: Optional[AgentSetup],
) -> Optional[LLMClientConfiguration]:
    if agent_setup is None or not agent_setup.llm_configuration_name:
        return None
    return agent_setup_retriever.llm_configuration_store.get(
        agent_setup.llm_configuration_name
    )


def effective_llm_configuration_name(
    agent_setup: Optional[AgentSetup],
    agent_configuration: Optional[dict],
) -> Optional[str]:
    """The name the request actually runs on: the model policy in the agent
    configuration when the setup allows it, the setup's own name otherwise."""
    default = agent_setup.llm_configuration_name if agent_setup else None
    requested = (
        (agent_configuration or {}).get("llm_selection_configuration") or {}
    ).get("llm_configuration_name")
    if not requested or requested == default:
        return default
    allowed = (agent_setup.allowed_llm_configuration_names or []) if agent_setup else []
    return requested if requested in allowed else default


# TODO: Think of a better way of passing the api key to RAGElo
os.environ["OPENAI_API_KEY"] = next(
    iter(
        api_key
        for ags in asyncio.run(agent_setup_retriever.list())
        if (configuration := llm_configuration_for(ags))
        and configuration.vendor_configuration
        and (openai_conf := configuration.vendor_configuration.openai)
        and (api_key := openai_conf.openai_api_key.get_unencrypted_secret())
    ),
    "",
)

debug_storage: List[Any] = []
mcp_oauth_store, mcp_oauth_token_client = setup_local_mcp_oauth(AgentDependencyRegistry)
bootstrap = setup_bootstrap(
    agent_registries_factory=LocalAgentRegistriesFactory(
        agent_setup_retriever=agent_setup_retriever,
        chat_agent_class_registry=ChatAgentClassRegistry,
        agent_dependency_registry=AgentDependencyRegistry,
        llm_configuration_store=agent_setup_retriever.llm_configuration_store,
    ),
    debug_backend=debug_storage.append,
    mcp_oauth_store=mcp_oauth_store,
    mcp_oauth_token_client=mcp_oauth_token_client,
    agent_state_store=agent_state_store,
)

message_bus = bootstrap.message_bus
TO_CHAT_MESSAGE_NAME = {
    ChatMessageSender.USER: "user",
    ChatMessageSender.BOT: "assistant",
}


def get_query_param(name: str) -> Optional[str]:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def get_request_path() -> str:
    url = getattr(st.context, "url", "")
    return urlparse(url).path if url else ""


def get_evidence_link_url(evidence_url: str) -> Optional[str]:
    document_id = extract_internal_document_id_from_evidence_url(evidence_url)
    if document_id is not None:
        return f"{zav_fe_url}/documents/{document_id}_0"

    parsed_url = urlparse(evidence_url)
    if parsed_url.scheme in {"http", "https"}:
        return evidence_url

    return None


def handle_mcp_oauth_callback():
    state = get_query_param("state")
    code = get_query_param("code")
    error = get_query_param("error")
    request_path = get_request_path()
    if not state or (not code and not error):
        return
    if request_path not in {"", "/", MCP_OAUTH_CALLBACK_PATH}:
        return

    try:
        results = asyncio.run(
            message_bus.handle(
                commands.HandleMCPOAuthCallback(
                    state=state,
                    code=code,
                    error=error,
                )
            )
        )
        result = results.pop(0)
    except Exception as exception:
        st.error(f"Authorization failed: {exception}")
        st.stop()

    if result.status == "error":
        st.error("Authorization failed. You can close this window and try again.")
    else:
        st.success("Authorization complete. You can close this window.")
    st.stop()


handle_mcp_oauth_callback()


def get_request_tenant(agent_configuration: dict) -> str:
    return agent_configuration.get("tenant") or local_tenant


def get_request_headers() -> RequestHeaders:
    return RequestHeaders(requester_uuid=local_requester_uuid)


def new_trace_file_name(agent_identifier: str):
    time_now = str(datetime.now().isoformat())
    trace_file_name = (
        f"{storage_path}/agent-traces/{time_now}_{agent_identifier}_trace.json"
    )
    st.session_state.trace_file_name = trace_file_name
    return trace_file_name


def get_trace_file_names():
    base_prefix = f"{storage_path}/agent-traces/"
    object_attributes = asyncio.run(
        object_storage_repo.filter_objects_attributes(
            url_prefix=f"{storage_path}/agent-traces"
        )
    )
    trace_file_names = []
    for object_attribute in object_attributes:
        if not object_attribute.url.endswith(".json"):
            continue
        relative_path = object_attribute.url.split(base_prefix, 1)[-1]
        if "/" in relative_path:
            # Skip nested folders such as the resumable agent state store
            continue
        trace_file_names.append(relative_path)
    return sorted(trace_file_names)


def store_trace_file_content(agent_identifier: str, entries: List[ChatEntry]):
    if "trace_file_name" not in st.session_state:
        trace_file_name = new_trace_file_name(agent_identifier=agent_identifier)
    else:
        trace_file_name = st.session_state.trace_file_name

    content = TraceFileContent.from_entries(entries=entries)
    if PYDANTIC_V2:
        text = content.model_dump_json(indent=2)
    else:
        text = content.json(indent=2)
    payload_bytes = text.encode("utf-8")

    asyncio.run(
        object_storage_repo.add(
            ObjectStorageItem(
                url=trace_file_name,
                payload=payload_bytes,
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


def load_previous_conversation():
    selected_trace_file_name = st.session_state.get("existing_trace_select")
    if not selected_trace_file_name:
        return
    trace_file_content = retrieve_trace_file_content(selected_trace_file_name)
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
    else:
        if tool.display_text:
            return tool.display_text
        msg = f"Running {tool.name}..."
        if tool.status == "completed":
            msg = f"Completed {tool.name}"
        elif tool.status == "error":
            msg = f"Error in {tool.name}"
    return msg


def get_mcp_oauth_action(tool: ContentPartTool) -> Optional[dict]:
    response = tool.response or {}
    if response.get("status") != "authorization_required":
        return None
    if not response.get("authorization_url"):
        return None
    return response


def render_mcp_oauth_action(st_elem, action: dict):
    label = html.escape(action.get("action_label") or "Authorize connection")
    authorization_url = html.escape(str(action["authorization_url"]), quote=True)
    st_elem.markdown(
        (
            f'<a href="{authorization_url}" target="_blank" '
            'rel="noopener noreferrer">'
            f"{label}</a>"
        ),
        unsafe_allow_html=True,
    )


def get_tool_image_uri(tool: ContentPartTool) -> Optional[str]:
    if not isinstance(tool.response, dict):
        return None
    image_uri = tool.response.get("image_uri")
    if isinstance(image_uri, str) and image_uri.startswith("data:image/"):
        return image_uri
    return None


# Render charts as a contained figure (like the web's capped, centered image)
# instead of full-bleed, so a multi-chart answer doesn't become a wall of images.
PLOT_DISPLAY_WIDTH = 720


def render_image_uri(st_elem, image_uri: str, caption: str = "Generated plot") -> None:
    try:
        if image_uri.startswith("data:image/"):
            _, encoded = image_uri.split(",", 1)
            image_data = base64.b64decode(encoded)
            st_elem.image(image_data, caption=caption, width=PLOT_DISPLAY_WIDTH)
        else:
            st_elem.image(image_uri, caption=caption, width=PLOT_DISPLAY_WIDTH)
    except Exception as e:
        st_elem.error(f"Failed to display image: {str(e)}")


# `create_plot` embeds charts as `![caption](plot:<id>)` and ships the bytes on
# the tool response; resolve those inline (st.markdown can't render `plot:`).
_PLOT_MARKER = re.compile(r"!\[([^\]]*)\]\(plot:([^)\s]+)\)")


def collect_plot_images(content: ChatMessage):
    keyed: Dict[str, str] = {}
    loose: List[str] = []
    for content_part in content.content_parts or []:
        if content_part.type != "tool" or not content_part.tool:
            continue
        image_uri = get_tool_image_uri(content_part.tool)
        if not image_uri:
            continue
        response = content_part.tool.response or {}
        plot_id = response.get("plot_id") if isinstance(response, dict) else None
        if plot_id:
            keyed[str(plot_id)] = image_uri
        else:
            # e.g. dataframe plot tools, which return an image but no plot_id.
            loose.append(image_uri)
    return keyed, loose


def apply_evidence_links(content: ChatMessage, text: str) -> str:
    if not content.evidences:
        return text
    seen_evidences = set()
    for evidence in content.evidences:
        if evidence.anchor_text and evidence.anchor_text not in seen_evidences:
            seen_evidences.add(evidence.anchor_text)
            evidence_link_url = get_evidence_link_url(evidence.document_hit_url)
            if evidence_link_url is None:
                continue
            text = text.replace(
                evidence.anchor_text,
                f"[{evidence.anchor_text}]({evidence_link_url})",
            )
    return text


def render_markdown_with_plots(
    render_target, text: str, keyed_plots: Dict[str, str]
) -> None:
    # Render markdown, replacing each `![caption](plot:<id>)` marker with its
    # chart inline. Charts only appear at their marker, so there is no image dump.
    cursor = 0
    for match in _PLOT_MARKER.finditer(text):
        before = text[cursor : match.start()]
        cursor = match.end()
        if before.strip():
            render_target.markdown(before)
        caption, plot_id = match.group(1), match.group(2)
        image_uri = keyed_plots.get(plot_id)
        if image_uri:
            render_image_uri(render_target, image_uri, caption or "Generated plot")
        elif caption.strip():
            render_target.markdown(caption)
    remainder = text[cursor:]
    if remainder.strip():
        render_target.markdown(remainder)


def render_tool_indicator_block(render_target, lines: List[str]) -> None:
    block = "\n\n".join(line for line in lines if line)
    if block.strip():
        render_target.markdown(block)


def render_chat_message_item_content(st_elem, content: ChatMessage):
    parts = content.content_parts or []
    keyed_plots, loose_plots = collect_plot_images(content)
    render_target = st_elem.container()
    mcp_oauth_actions: List[dict] = []
    pending_tool_lines: List[str] = []
    rendered_text = False

    # Render content parts in document order (web parity): the agent's text
    # before/between tool calls is preserved, consecutive tool calls render as
    # one grouped indicator block, and each chart is spliced into the prose at
    # its `![caption](plot:<id>)` marker.
    for part in parts:
        if part.type == "tool" and part.tool:
            pending_tool_lines.append(render_tool_content(part.tool))
            mcp_oauth_action = get_mcp_oauth_action(part.tool)
            if mcp_oauth_action:
                mcp_oauth_actions.append(mcp_oauth_action)
            continue
        render_tool_indicator_block(render_target, pending_tool_lines)
        pending_tool_lines = []
        if part.type == "text" and part.text:
            render_markdown_with_plots(
                render_target, apply_evidence_links(content, part.text), keyed_plots
            )
            rendered_text = True
        elif part.type == "table":
            render_markdown_with_plots(
                render_target,
                apply_evidence_links(content, content.content or ""),
                keyed_plots,
            )
            rendered_text = True
    render_tool_indicator_block(render_target, pending_tool_lines)

    # Agents that return the answer only on `content` (no text content parts).
    if not rendered_text and content.content:
        render_markdown_with_plots(
            render_target, apply_evidence_links(content, content.content), keyed_plots
        )

    for action in mcp_oauth_actions:
        render_mcp_oauth_action(render_target, action)

    # Dataframe plot tools return an image but no plot_id / marker, so there is
    # nothing to splice them into; render them on their own, as the web's
    # PlotImageRenderer does at the tool node.
    for image_uri in loose_plots:
        render_image_uri(render_target, image_uri)
    if hasattr(content, "image_uri") and content.image_uri:
        render_image_uri(render_target, content.image_uri)
    return render_target


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
    stateful = bool(options.get("stateful_conversation", False))
    session_id = options.get("session_id")
    session_id = session_id if isinstance(session_id, str) else None

    async def compute_conversation():
        agent_setup = compute_chat_message_item.chat_configuration_item.agent_setup
        agent_configuration = agent_setup.agent_configuration or {}
        command_kwargs: dict[str, Any] = dict(
            tenant=get_request_tenant(agent_configuration),
            index_id=agent_configuration.get("index_id", None),
            request_headers=get_request_headers(),
            chat_request=ChatRequest(
                agent_identifier=agent_setup.agent_identifier,
                conversation=conversation,
                conversation_context=conversation_context,
            ),
            stateful=stateful,
        )
        if session_id:
            command_kwargs["session_id"] = session_id
        command = commands.CreateChatResponse(**command_kwargs)
        results = await compute_chat_message_item.message_bus.handle(command)
        return results.pop(0)

    async def compute_conversation_streaming():
        agent_setup = compute_chat_message_item.chat_configuration_item.agent_setup
        agent_configuration = agent_setup.agent_configuration or {}
        command_kwargs: dict[str, Any] = dict(
            tenant=get_request_tenant(agent_configuration),
            index_id=agent_configuration.get("index_id", None),
            request_headers=get_request_headers(),
            chat_request=ChatRequest(
                agent_identifier=agent_setup.agent_identifier,
                conversation=conversation,
                conversation_context=conversation_context,
            ),
            stateful=stateful,
        )
        if session_id:
            command_kwargs["session_id"] = session_id
        command = commands.CreateChatStream(**command_kwargs)
        return compute_chat_message_item.message_bus.handle_stream(command)

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
        last_debug_write = 0.0
        while True:
            try:
                conversation_task = asyncio.create_task(anext(conversation_generator))
                while not conversation_task.done():
                    now = time.monotonic()
                    if (
                        agent_debug_logs
                        and debug_storage
                        and now - last_debug_write >= 1.0
                    ):
                        agent_debug_logs.json(debug_storage)
                        last_debug_write = now

                    await asyncio.sleep(0.1)
                conversation_result = conversation_task.result()
                last_message = conversation_result
                render_chat_message_item_content(
                    chat_message_item_content, last_message
                )
            except StopAsyncIteration:
                break
            except Exception as e:
                chat_message_item_content.error(f"Agent error: {e}")
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
            effective_name = effective_llm_configuration_name(
                chat_configuration_item.agent_setup, agent_configuration
            )
            llm_client_configuration = (
                agent_setup_retriever.llm_configuration_store.get(effective_name)
                if effective_name
                else None
            )
            llm_model_configuration = (
                llm_client_configuration.model_configuration
                if llm_client_configuration
                else None
            )
            st.caption("Configuration")
            st.text(effective_name)
            st.caption("Vendor")
            st.text(
                llm_client_configuration.vendor.value
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
            st.caption("Reasoning effort")
            st.text(
                llm_model_configuration.reasoning_effort
                if llm_model_configuration
                else None
            )

            st.subheader("Conversation context")
            cc = chat_configuration_item.conversation_context
            if cc:
                if cc.tag_context:
                    st.caption("Tag IDs")
                    st.json(json.dumps(cc.tag_context.tag_ids, indent=2))
                if cc.document_context:
                    st.caption("Document IDs")
                    st.json(json.dumps(cc.document_context.document_ids, indent=2))
                    st.caption("Retrieval unit")
                    st.text(cc.document_context.retrieval_unit)
                if cc.custom_context:
                    st.caption("Custom context items")
                    if PYDANTIC_V2:
                        st.json(cc.custom_context.model_dump_json(indent=2))
                    else:
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


def get_previous_response_session_id(
    previous_chat_messages: List[ChatMessageItem],
) -> Optional[str]:
    return next(
        (
            chat_message_item.message.session_id
            for chat_message_item in reversed(previous_chat_messages)
            if chat_message_item.message.sender == ChatMessageSender.BOT
            and chat_message_item.message.session_id
        ),
        None,
    )


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
        chat_message_item_content = st.empty()
        chat_message_item_content.markdown("Thinking...")
        messages = [chat_message.message for chat_message in previous_chat_messages]
        # Echo the conversation's session id back in every mode so the traces
        # of all turns stay grouped; only stateful mode also trims the history
        # to the new turn (the session transcript is restored server-side).
        session_id = get_previous_response_session_id(previous_chat_messages)
        if options.get("stateful_conversation") and messages:
            messages = [messages[-1]]

        created_chat_response = asyncio.run(
            create_chat_response(
                compute_chat_message_item,
                messages,
                conversation_context,
                chat_message_item_content,
                chat_message_item_debug_panel_status,
                session_id=session_id,
                **options,
            )
        )
        return created_chat_response


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
            entries.append(ChatEntry.from_message(created_chat_message_item))
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
        if created_chat_message_item:
            entries.append(ChatEntry.from_message(created_chat_message_item))


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
        st.session_state["existing_trace_select"] = None
        start_new_conversation()

    st.selectbox(
        "Load previous conversation",
        get_trace_file_names(),
        index=None,
        key="existing_trace_select",
        on_change=load_previous_conversation,
    )

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
            ),
            st.session_state.agent_identifier,
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
        try:
            edited_agent_configuration = (
                json.loads(sel_agent_configuration) if sel_agent_configuration else {}
            )
        except json.JSONDecodeError:
            edited_agent_configuration = agent_configuration
        effective_name = effective_llm_configuration_name(
            agent_setup, edited_agent_configuration
        )
        name_options = [
            name
            for name in agent_setup_retriever.llm_configuration_store.names()
            if not name.startswith("__agent__:")
        ]
        if effective_name and effective_name not in name_options:
            name_options.insert(0, effective_name)
        sel_llm_configuration_name = st.selectbox(
            "Configuration",
            name_options,
            index=(
                name_options.index(effective_name)
                if effective_name in name_options
                else None
            ),
            help=(
                "The named LLM configuration the agent runs on. "
                "The fields below edit the selected configuration."
            ),
        )
        llm_client_configuration = (
            agent_setup_retriever.llm_configuration_store.get(
                sel_llm_configuration_name
            )
            if sel_llm_configuration_name
            else None
        )
        llm_model_configuration = (
            llm_client_configuration.model_configuration
            if llm_client_configuration
            else None
        )
        vendor_options = [e.value for e in LLMProviderName]
        sel_vendor = st.selectbox(
            "Vendor",
            vendor_options,
            index=(
                vendor_options.index(llm_client_configuration.vendor.value)
                if llm_client_configuration
                else None
            ),
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
        sel_reasoning_effort = st.text_input(
            "Reasoning effort",
            value=(
                llm_model_configuration.reasoning_effort
                if llm_model_configuration and llm_model_configuration.reasoning_effort
                else ""
            ),
            help=(
                "Constrains effort on reasoning for reasoning models "
                "(e.g. low, medium, high)"
            ),
        )

    conversation_context: Optional[ConversationContext] = None
    with st.expander("Conversation context"):
        st.caption("Tag context")
        sel_tag_ids = st.text_area(
            "Tag IDs",
            json.dumps([], indent=2),
            help="List of tag IDs to use as context",
        )

        st.caption("Document context")
        sel_document_ids = st.text_area(
            "Document IDs",
            json.dumps([], indent=2),
            help="Can be combined with tag context",
        )
        sel_retrieval_unit = st.text_input("Retrieval unit")

        st.caption("Custom context")
        sel_custom_context_items = st.text_area(
            "Items",
            json.dumps([], indent=2),
            help="Can be combined with tag context (mutually exclusive with document context)",  # noqa: E501
        )

    # Build conversation context based on selections
    tag_context = None
    document_context = None
    custom_context = None

    try:
        tag_ids_list = json.loads(sel_tag_ids) if sel_tag_ids else []
        if tag_ids_list:
            tag_context = TagContext(tag_ids=tag_ids_list)
    except json.JSONDecodeError:
        st.error("Invalid JSON in Tag IDs")

    try:
        doc_ids_list = json.loads(sel_document_ids) if sel_document_ids else []
        if doc_ids_list and sel_retrieval_unit:
            document_context = DocumentContext(
                document_ids=doc_ids_list,
                retrieval_unit=sel_retrieval_unit,
            )
    except json.JSONDecodeError:
        st.error("Invalid JSON in Document IDs")

    try:
        custom_items_list = (
            json.loads(sel_custom_context_items) if sel_custom_context_items else []
        )
        if custom_items_list:
            custom_context = CustomContext(
                items=[CustomContextItem(**item) for item in custom_items_list]
            )
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"Invalid JSON in Custom Context Items: {e}")

    if tag_context or document_context or custom_context:
        conversation_context = ConversationContext(
            tag_context=tag_context,
            document_context=document_context,
            custom_context=custom_context,
        )
    else:
        conversation_context = None

    st.session_state.conversation_context = conversation_context

    with st.expander("UI configuration"):
        sel_print_debug_logs = st.toggle("Show debug logs", value=True)
        sel_streaming_mode = st.toggle("Streaming mode", value=True)
        sel_stateful_conversation = st.toggle("Stateful conversation", value=True)

    with st.expander("Conversation context for next message", expanded=False):
        st.caption("Leave empty to use Agent Instance level context")

        st.text_area(
            "Tag IDs (per message)",
            help="Add tag context for this message only",
            key="msg_tag_ids",
        )
        st.text_area(
            "Document IDs (per message)",
            help="Add document context for this message only",
            key="msg_document_ids",
        )
        st.text_input(
            "Retrieval unit (per message)",
            key="msg_retrieval_unit",
        )
        st.text_area(
            "Custom context items (per message)",
            help="Add custom context for this message only",
            key="msg_custom_context_items",
        )

    if agent_setup:
        if sel_agent_configuration:
            agent_setup.agent_configuration = json.loads(sel_agent_configuration)
        current_llm_configuration = (
            agent_setup_retriever.llm_configuration_store.get(
                sel_llm_configuration_name
            )
            if sel_llm_configuration_name
            else None
        )
        if current_llm_configuration and sel_llm_configuration_name:
            updated_llm_configuration = current_llm_configuration.copy(deep=True)
            if sel_vendor:
                updated_llm_configuration.vendor = LLMProviderName(sel_vendor)
            if sel_name:
                updated_llm_configuration.model_configuration.name = sel_name
            if sel_type:
                updated_llm_configuration.model_configuration.type = LLMModelType(
                    sel_type
                )
            if sel_temperature:
                updated_llm_configuration.model_configuration.temperature = (
                    sel_temperature
                )
            if sel_max_tokens:
                updated_llm_configuration.model_configuration.max_tokens = int(
                    sel_max_tokens
                )
            updated_llm_configuration.model_configuration.reasoning_effort = (
                sel_reasoning_effort if sel_reasoning_effort else None
            )
            agent_setup_retriever.llm_configuration_store.set(
                sel_llm_configuration_name, updated_llm_configuration
            )
            # The picker drives the run unless the agent configuration pins a
            # model policy, which is the stronger channel.
            policy_name = (
                (agent_setup.agent_configuration or {}).get(
                    "llm_selection_configuration"
                )
                or {}
            ).get("llm_configuration_name")
            if not policy_name:
                agent_setup.llm_configuration_name = sel_llm_configuration_name
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


if (  # noqa: C901
    "agent_identifier" not in st.session_state or "agent_setup" not in st.session_state
):
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
            stateful_conversation=sel_stateful_conversation,
        )

    def clear_per_message_context():
        """Callback to clear per-message context after sending."""
        # Per-message context
        msg_tag_ids = st.session_state.get("msg_tag_ids", "")
        msg_document_ids = st.session_state.get("msg_document_ids", "")
        msg_retrieval_unit = st.session_state.get("msg_retrieval_unit", "")
        msg_custom_context_items = st.session_state.get("msg_custom_context_items", "")
        # Build per-message context if provided
        st.session_state.message_context = None
        if any([msg_tag_ids, msg_document_ids, msg_custom_context_items]):
            msg_tag_ctx = None
            msg_doc_ctx = None
            msg_custom_ctx = None

            try:
                if msg_tag_ids:
                    msg_tag_list = json.loads(msg_tag_ids)
                    if msg_tag_list:
                        msg_tag_ctx = TagContext(tag_ids=msg_tag_list)
            except json.JSONDecodeError:
                st.error("Invalid JSON in per-message Tag IDs")

            try:
                if msg_document_ids:
                    msg_doc_list = json.loads(msg_document_ids)
                    if msg_doc_list and msg_retrieval_unit:
                        msg_doc_ctx = DocumentContext(
                            document_ids=msg_doc_list,
                            retrieval_unit=msg_retrieval_unit,
                        )
            except json.JSONDecodeError:
                st.error("Invalid JSON in per-message Document IDs")

            try:
                if msg_custom_context_items:
                    msg_custom_list = json.loads(msg_custom_context_items)
                    if msg_custom_list:
                        msg_custom_ctx = CustomContext(
                            items=[
                                CustomContextItem(**item) for item in msg_custom_list
                            ]
                        )
            except (json.JSONDecodeError, TypeError) as e:
                st.error(f"Invalid JSON in per-message Custom Context: {e}")

            if msg_tag_ctx or msg_doc_ctx or msg_custom_ctx:
                st.session_state.message_context = ConversationContext(
                    tag_context=msg_tag_ctx,
                    document_context=msg_doc_ctx,
                    custom_context=msg_custom_ctx,
                )

        st.session_state.msg_tag_ids = ""
        st.session_state.msg_document_ids = ""
        st.session_state.msg_retrieval_unit = ""
        st.session_state.msg_custom_context_items = ""

    if content := st.chat_input(
        "What is up?", accept_file="multiple", on_submit=clear_per_message_context
    ):
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
                entry=ChatEntry.from_configuration(chat_configuration_item),
                print_debug_logs=sel_print_debug_logs,
                streaming_mode=sel_streaming_mode,
                stateful_conversation=sel_stateful_conversation,
            )

        # Build content_parts for the user message
        content_parts = []

        if st.session_state.message_context:
            content_parts.append(
                ContentPart(type="context", context=st.session_state.message_context)
            )

        # Always add text content
        content_parts.append(ContentPart(type="text", text=content.text))

        # Handle file attachments (only images for now)
        image_uri: Optional[str] = None
        if content.files:
            for file in content.files:
                # Check if file is an image
                if file.type and file.type.startswith("image/"):
                    # Read file bytes and convert to base64 data URL
                    file_bytes = file.read()
                    encoded = base64.b64encode(file_bytes).decode("utf-8")
                    image_uri = f"data:{file.type};base64,{encoded}"
                else:
                    # TODO
                    pass
        render_entry(
            entries=st.session_state.entries,
            entry=ChatEntry.from_message(
                ChatMessageItem(
                    message=ChatMessage(
                        sender=ChatMessageSender.USER,
                        content=content.text,
                        content_parts=content_parts,
                        image_uri=image_uri,
                    )
                )
            ),
            print_debug_logs=sel_print_debug_logs,
            streaming_mode=sel_streaming_mode,
            stateful_conversation=sel_stateful_conversation,
        )

        render_entry(
            entries=st.session_state.entries,
            compute_message_item=ComputeChatMessageItem(
                message_bus=message_bus,
                chat_configuration_item=chat_configuration_item,
            ),
            conversation_context=st.session_state.conversation_context,
            print_debug_logs=sel_print_debug_logs,
            streaming_mode=sel_streaming_mode,
            stateful_conversation=sel_stateful_conversation,
        )
        store_trace_file_content(
            agent_identifier=st.session_state.agent_identifier,
            entries=st.session_state.entries,
        )
