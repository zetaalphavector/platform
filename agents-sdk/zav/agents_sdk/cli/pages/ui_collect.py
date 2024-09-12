import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from pydantic import BaseModel
from ragelo import Query
from ragelo.types.configurations import (
    CustomPromptAnswerEvaluatorConfig,
    EloAgentRankerConfig,
    PairwiseEvaluatorConfig,
    ReasonerEvaluatorConfig,
)
from zav.object_storage_repo import ObjectStorageItem

from zav.agents_sdk import ChatMessage, ChatMessageSender
from zav.agents_sdk.cli.ui_app import (
    ChatConfigurationItem,
    ChatEntry,
    ChatMessageItem,
    ComputeChatMessageItem,
    agent_setup_retriever,
    message_bus,
    object_storage_repo,
    render_chat_configuration_item,
    render_entry,
    start_new_conversation,
    storage_path,
    store_trace_file_content,
)

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


class EvaluationAnswerTrace(BaseModel):
    agent_hash: str
    qid: str
    trace_file_name: str


class RageloLLMConfig(BaseModel):
    llm_provider: str
    model_name: str
    max_tokens: int


class RageloEvaluation(BaseModel):
    llm_config: RageloLLMConfig
    reasoner_config: ReasonerEvaluatorConfig
    pairwise_config: PairwiseEvaluatorConfig
    custom_agent_eval_config: CustomPromptAnswerEvaluatorConfig
    elo_ranker_config: EloAgentRankerConfig
    queries: List[Query]
    elo_rankings: Optional[Dict[str, int]] = None


class EvaluationFileContent(BaseModel):
    agent_configurations: List[ChatConfigurationItem]
    queries: Dict[str, str]
    answer_traces: List[EvaluationAnswerTrace]
    ragelo: Optional[RageloEvaluation] = None


def new_eval_file_name():
    time_now = str(datetime.now().isoformat())
    eval_file_name = f"{storage_path}/agent-evals/{time_now}_eval.json"
    st.session_state.eval_file_name = eval_file_name
    return eval_file_name


def get_eval_file_names():
    object_attributes = asyncio.run(
        object_storage_repo.filter_objects_attributes(
            url_prefix=f"{storage_path}/agent-evals"
        )
    )

    eval_file_names: List[str] = []
    for object_attribute in object_attributes:
        if object_attribute.url.endswith(".json"):
            eval_file_names.append(object_attribute.url.split("/")[-1])
    return sorted(eval_file_names)


def store_eval_file_content(evaluation_file_content: EvaluationFileContent):
    if "eval_file_name" not in st.session_state:
        eval_file_name = new_eval_file_name()
    else:
        eval_file_name = st.session_state.eval_file_name

    if not getattr(st.session_state, "eval_file_name2content", {}):
        st.session_state.eval_file_name2content = defaultdict()
    st.session_state.eval_file_name2content[eval_file_name] = evaluation_file_content

    asyncio.run(
        object_storage_repo.add(
            ObjectStorageItem(
                url=eval_file_name,
                payload=evaluation_file_content.json(indent=2).encode("utf-8"),
            )
        )
    )


def retrieve_eval_file_content(eval_file_name: str) -> EvaluationFileContent:
    if not getattr(st.session_state, "eval_file_name2content", {}):
        st.session_state.eval_file_name2content = defaultdict()
    try:
        url_prefix = f"{storage_path}/agent-evals"
        if eval_file_name in st.session_state.eval_file_name2content:
            return st.session_state.eval_file_name2content[eval_file_name]

        eval_file_content_obj = asyncio.run(
            object_storage_repo.get(
                f"{url_prefix}/{eval_file_name.replace('agent-evals/', '')}"
                if not eval_file_name.startswith(url_prefix)
                else eval_file_name
            )
        )
        if not eval_file_content_obj:
            return EvaluationFileContent(
                agent_configurations=[], queries={}, answer_traces=[]
            )
        eval_file_content = EvaluationFileContent.parse_raw(
            eval_file_content_obj.payload
        )
        st.session_state.eval_file_name2content[eval_file_name] = eval_file_content
        return eval_file_content
    except FileNotFoundError:
        return EvaluationFileContent(
            agent_configurations=[], queries={}, answer_traces=[]
        )


def __get_existing_trace_file(agent_hash: str, qid: str, query: str) -> Optional[str]:
    for eval_file_name in get_eval_file_names():
        eval_file_content = retrieve_eval_file_content(eval_file_name)
        for eval_trace in eval_file_content.answer_traces:
            if (
                eval_trace.agent_hash == agent_hash
                and eval_trace.qid == qid
                and (eval_file_content.queries[qid] == query)
            ):
                return eval_trace.trace_file_name
    return None


st.logo(
    "https://search.zeta-alpha.com/assets/img/zeta-logo-full-white.svg",
    link="https://docs.zeta-alpha.com/gen-ai/customize/getting-started",
    icon_image="https://search.zeta-alpha.com/assets/img/zeta-logo-white.svg",
)


st.sidebar.page_link("ui_app.py", label="Chat", icon="💬")
st.sidebar.page_link("pages/ui_collect.py", label="Collect", icon="🪣")
st.sidebar.page_link("pages/ui_eval.py", label="RAGElo Evaluation", icon="🔎")
sel_existing_eval: Optional[str] = st.sidebar.selectbox(
    "Load previous evaluation", get_eval_file_names(), index=None
)
if sel_existing_eval:
    eval_file_content = retrieve_eval_file_content(sel_existing_eval)
    st.session_state.eval_chat_config_items = {
        chat_configuration_item.hash(): chat_configuration_item
        for chat_configuration_item in eval_file_content.agent_configurations
    }
    st.session_state.eval_chat_queries = eval_file_content.queries
    st.session_state.eval_answer_traces = eval_file_content.answer_traces

    st.session_state.eval_file_name = f"{storage_path}/agent-evals/{sel_existing_eval}"
else:
    st.session_state.eval_answer_traces = []
    st.session_state.eval_file_name = None

st.subheader("Selected agents")
eval_chat_config_items: Dict[str, ChatConfigurationItem] = getattr(
    st.session_state, "eval_chat_config_items", {}
)
if eval_chat_config_items:
    for chat_config_hash, chat_config_item in eval_chat_config_items.items():
        render_chat_configuration_item(
            chat_config_item, True, chat_configuration_key_postfix=chat_config_hash
        )
else:
    st.warning("No agents selected.")

current_eval_file_name = sel_existing_eval or new_eval_file_name()

st.subheader("Selected queries")
existing_chat_queries = "\n".join(
    getattr(st.session_state, "eval_chat_queries", {}).values()
)
queries: str = st.text_area(
    "",
    placeholder="Enter each query on a new line, no quotes.",
    value=existing_chat_queries or "",
)
eval_chat_queries = queries.split("\n")
queries_df = pd.DataFrame(
    [
        {"qid": str(idx + 1), "query": query}
        for idx, query in enumerate(eval_chat_queries)
    ]
)
edited_queries_df = st.data_editor(queries_df, num_rows="dynamic")
st.session_state.eval_chat_queries = {
    row["qid"]: row["query"] for row in edited_queries_df.to_dict(orient="records")
}

run_cases = {
    f'{chat_config_hash}-{row["qid"]}': {
        "chat_configuration_item": chat_configuration_item,
        "row": row,
        "trace_file_name": __get_existing_trace_file(
            chat_config_hash, row["qid"], row["query"]
        ),
    }
    for row in edited_queries_df.to_dict(orient="records")
    for chat_config_hash, chat_configuration_item in eval_chat_config_items.items()
}


if run_cases:
    if st.button("Collect answers"):
        for (
            chat_configuration_hash,
            chat_configuration_item,
        ) in eval_chat_config_items.items():
            st.session_state.agent_identifier = chat_configuration_item.agent_identifier
            agent_setup_retriever.update_agent_setup(
                agent_identifier=st.session_state.agent_identifier,
                agent_setup_patch=chat_configuration_item.agent_setup.dict(
                    exclude_unset=True,
                    exclude_none=True,
                    exclude_defaults=True,
                ),
            )
            for row in edited_queries_df.to_dict(orient="records"):
                run_case_id = f"{chat_configuration_hash}-{row['qid']}"
                if existing_trace_file_name := run_cases[run_case_id][
                    "trace_file_name"
                ]:
                    st.markdown(f"qid: {row['qid']}, query: {row['query']}")
                    st.markdown(f"trace file: {existing_trace_file_name} ✅")
                    trace_file_name = existing_trace_file_name
                else:
                    st.markdown(f"qid: {row['qid']}, query: {row['query']}")
                    start_new_conversation()
                    trace_file_name = st.session_state.trace_file_name
                    st.caption(f"Saving to trace file: {trace_file_name}")
                    render_entry(
                        entries=st.session_state.entries,
                        entry=ChatEntry(
                            chat_configuration_item=chat_configuration_item
                        ),
                        print_debug_logs=True,
                        render_expander_title=True,
                        chat_configuration_key_postfix=chat_configuration_hash
                        + f"qid{row['qid']}",
                    )
                    render_entry(
                        entries=st.session_state.entries,
                        entry=ChatEntry(
                            chat_message_item=ChatMessageItem(
                                message=ChatMessage(
                                    sender=ChatMessageSender.USER,
                                    content=row["query"],
                                )
                            )
                        ),
                        print_debug_logs=True,
                    )
                    try:
                        render_entry(
                            entries=st.session_state.entries,
                            compute_message_item=ComputeChatMessageItem(
                                message_bus=message_bus,
                                chat_configuration_item=chat_configuration_item,
                            ),
                            conversation_context=None,
                            print_debug_logs=True,
                        )

                        store_trace_file_content(
                            agent_identifier=st.session_state.agent_identifier,
                            entries=st.session_state.entries,
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                        continue

                current_eval_file_content = retrieve_eval_file_content(
                    current_eval_file_name
                )
                run_cases[f"{chat_configuration_hash}-{row['qid']}"][
                    "trace_file_name"
                ] = trace_file_name

                store_eval_file_content(
                    EvaluationFileContent(
                        agent_configurations=list(eval_chat_config_items.values()),
                        queries=getattr(st.session_state, "eval_chat_queries", {}),
                        answer_traces=current_eval_file_content.answer_traces
                        + [
                            EvaluationAnswerTrace(
                                agent_hash=chat_configuration_hash,
                                qid=row["qid"],
                                trace_file_name=trace_file_name,
                            )
                        ],
                        ragelo=current_eval_file_content.ragelo,
                    )
                )
