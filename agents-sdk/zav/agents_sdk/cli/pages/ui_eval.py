import hashlib
import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from ragelo import AgentAnswer, Document, Query
from ragelo.types.configurations import (
    CustomPromptAnswerEvaluatorConfig,
    EloAgentRankerConfig,
    PairwiseEvaluatorConfig,
    ReasonerEvaluatorConfig,
)
from zav.llm_domain import LLMClientConfiguration
from zav.logging import logger
from zav.prompt_completion import ChatClientFactory
from zav.pydantic_compat import PYDANTIC_V2

from zav.agents_sdk.adapters.llm_models.ragelo_llm_provider import (
    ZAVRageloLLMProvider,
)
from zav.agents_sdk.cli.models import (
    EvaluationFileContent,
    RageloEvaluation,
)
from zav.agents_sdk.cli.ragelo_evaluation import (
    DEFAULT_CUSTOM_PROMPT_CONFIG,
    DEFAULT_ELO_RANKER_CONFIG,
    DEFAULT_PAIRWISE_CONFIG,
    DEFAULT_REASONER_CONFIG,
    judge,
    rank_agents,
)
from zav.agents_sdk.cli.ui_app import (
    ChatEntry,
    ChatMessageItem,
    EvaluatorItem,
    agent_display_name,
    get_eval_file_names,
    llm_configuration_for,
    render_chat_configuration_item,
    render_entry,
    retrieve_eval_file_content,
    retrieve_trace_file_content,
    start_new_conversation,
    storage_path,
    store_eval_file_content,
)
from zav.agents_sdk.domain.chat_message import ChatMessage, ChatMessageSender


def format_model_json(model: Any) -> str:
    """
    Render a Pydantic model as a JSON string in a version-compatible way.
    """
    if PYDANTIC_V2:
        return model.model_dump_json(indent=2)
    return model.json(indent=2)


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


def __hashed(content: str):
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def __construct_ragelo_queries(eval_file_content: EvaluationFileContent) -> List[Query]:
    qid_to_query = (
        {
            ragelo_query.qid: ragelo_query
            for ragelo_query in eval_file_content.ragelo.queries
        }
        if eval_file_content.ragelo
        else {}
    )
    for qid, query in eval_file_content.queries.items():
        if qid not in qid_to_query:
            qid_to_query[qid] = Query(qid=qid, query=query)

    for answer_trace in eval_file_content.answer_traces:
        try:
            trace_content = retrieve_trace_file_content(answer_trace.trace_file_name)
        except FileNotFoundError:
            logger.exception(f"Trace file not found, {answer_trace.trace_file_name}")
            continue

        message_items = [
            entry.chat_message_item
            for entry in trace_content.entries
            if entry.chat_message_item
        ]
        if not message_items:
            logger.exception("No chat message items found in trace")
            continue
        last_message_item = message_items[-1]
        if not last_message_item.debug_storage:
            logger.exception("No debug storage found in chat message item")
            continue

        configuration_items = [
            entry.chat_configuration_item
            for entry in trace_content.entries
            if entry.chat_configuration_item
        ]
        last_configuration_item = configuration_items[-1]
        if not last_configuration_item:
            logger.exception("No configuration items found in trace")
            continue

        ragelo_query = qid_to_query[answer_trace.qid]
        llm_contexts = next(
            (
                d["llm_contexts"]
                for d in last_message_item.debug_storage
                if "llm_contexts" in d
            ),
            None,
        )
        evidences = last_message_item.message.evidences
        ragelo_agent = (
            f"{last_configuration_item.agent_identifier}-{answer_trace.agent_hash}"
        )
        ragelo_answer = last_message_item.message.content
        if llm_contexts:
            for llm_context in llm_contexts:
                content = (
                    llm_context["content"]
                    if isinstance(llm_context["content"], str)
                    else json.dumps(llm_context["content"])
                )
                ragelo_did = __hashed(content)
                ragelo_query.add_retrieved_doc(
                    Document(qid=ragelo_query.qid, did=ragelo_did, text=content),
                    exist_ok=True,
                )
                if evidences:
                    for evidence in evidences:
                        if (
                            (anchor_text := evidence.anchor_text)
                            and (citation_id := llm_context.get("citation_id"))
                            and citation_id in anchor_text
                        ):
                            ragelo_answer = ragelo_answer.replace(
                                anchor_text, f"[{ragelo_did}]"
                            )
                else:
                    citation_id = llm_context.get("citation_id")
                    if citation_id in ragelo_answer:
                        ragelo_answer = ragelo_answer.replace(citation_id, ragelo_did)

        elif evidences:
            for evidence in evidences:
                if not evidence.text_extract or not evidence.anchor_text:
                    continue
                ragelo_did = __hashed(evidence.text_extract)
                ragelo_answer = ragelo_answer.replace(
                    evidence.anchor_text, f"[{ragelo_did}]"
                )
                ragelo_query.add_retrieved_doc(
                    Document(
                        qid=ragelo_query.qid, did=ragelo_did, text=evidence.text_extract
                    ),
                    exist_ok=True,
                )
        else:
            logger.exception(
                "No documents or evidences found in trace "
                f"{answer_trace.trace_file_name}"
            )

        ragelo_query.add_agent_answer(
            AgentAnswer(qid=ragelo_query.qid, agent=ragelo_agent, text=ragelo_answer),
            exist_ok=True,
        )

    return list(qid_to_query.values())


def __config_text_area(
    label: str, value: Dict[str, Any], disabled: bool
) -> Dict[str, Any]:
    return json.loads(
        st.sidebar.text_area(
            label, value=json.dumps(value, indent=2), disabled=disabled
        )
    )


def __render_message(sender: ChatMessageSender, content: str):
    render_entry(
        entries=st.session_state.entries,
        entry=ChatEntry.from_message(
            ChatMessageItem(message=ChatMessage(sender=sender, content=content))
        ),
        print_debug_logs=False,
    )


def __render_query(query: Query):
    start_new_conversation()
    __render_message(ChatMessageSender.USER, query.query)


def __render_cited_documents(query: Query, *answers: AgentAnswer):
    for doc in query.retrieved_docs.values():
        if any(doc.did in answer.rendered_text for answer in answers):
            st.markdown(f"###### [{doc.did}]\n {doc.text}")


def __render_verdict(verdict: str, explanation: str):
    render_entry(
        entries=st.session_state.entries,
        entry=ChatEntry.from_evaluator(
            EvaluatorItem(verdict=verdict, explanation=explanation)
        ),
    )


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
ragelo = (
    retrieve_eval_file_content(sel_existing_eval).ragelo if sel_existing_eval else None
)

if ragelo:
    llm_config: Optional[LLMClientConfiguration] = LLMClientConfiguration.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "LLM Configuration",
                value=format_model_json(ragelo.llm_config),
                disabled=True,
            )
        )
    )
else:
    default_llm_config: Optional[LLMClientConfiguration] = None
    if sel_existing_eval:
        eval_file_content = retrieve_eval_file_content(sel_existing_eval)
        for agent_conf in eval_file_content.agent_configurations:
            if configuration := llm_configuration_for(agent_conf.agent_setup):
                default_llm_config = configuration
                break
    if default_llm_config:
        default_llm_json = format_model_json(default_llm_config)
    else:
        default_llm_json = "{}"
    llm_config_raw = st.sidebar.text_area(
        "LLM Configuration",
        value=default_llm_json,
    )
    try:
        llm_config = LLMClientConfiguration.parse_obj(json.loads(llm_config_raw))
    except Exception:
        llm_config = None
        st.sidebar.error(
            "Invalid LLM configuration. Please provide a valid JSON configuration."
        )

reasoner_config_raw = __config_text_area(
    "Reasoner Evaluator Configuration",
    ragelo.reasoner_config if ragelo else DEFAULT_REASONER_CONFIG,
    disabled=bool(ragelo),
)
custom_agent_eval_config_raw = __config_text_area(
    "Custom Prompt Answer Evaluator Configuration",
    ragelo.custom_agent_eval_config if ragelo else DEFAULT_CUSTOM_PROMPT_CONFIG,
    disabled=bool(ragelo),
)
pairwise_evaluator_config_raw = __config_text_area(
    "Pairwise Evaluator Configuration",
    ragelo.pairwise_config if ragelo else DEFAULT_PAIRWISE_CONFIG,
    disabled=bool(ragelo),
)
agent_ranker_config_raw = __config_text_area(
    "Elo Ranker Configuration",
    ragelo.elo_ranker_config if ragelo else DEFAULT_ELO_RANKER_CONFIG,
    disabled=False,
)
reasoner_config = ReasonerEvaluatorConfig.model_validate(reasoner_config_raw)
custom_agent_eval_config = CustomPromptAnswerEvaluatorConfig.model_validate(
    custom_agent_eval_config_raw
)
pairwise_evaluator_config = PairwiseEvaluatorConfig.model_validate(
    pairwise_evaluator_config_raw
)
agent_ranker_config = EloAgentRankerConfig.model_validate(agent_ranker_config_raw)

if sel_existing_eval:  # noqa
    eval_file_content = retrieve_eval_file_content(sel_existing_eval)
    st.session_state.eval_file_name = f"{storage_path}/agent-evals/{sel_existing_eval}"

    st.subheader("Selected agents")
    for conf in eval_file_content.agent_configurations:
        render_chat_configuration_item(conf, True)

    pairwise_clicked = st.button("Evaluate pairwise")
    pointwise_clicked = st.button("Evaluate pointwise")
    if pairwise_clicked or pointwise_clicked:
        if not llm_config:
            st.error(
                "Please provide a valid LLM configuration before running evaluation."
            )
            st.stop()
        if pairwise_clicked and len(eval_file_content.agent_configurations) < 2:
            st.error(
                "For a pairwise evaluation, please add at least two agents to "
                "the evaluator."
            )
            st.stop()
        assert llm_config is not None
        queries = __construct_ragelo_queries(eval_file_content)
        eval_llm_client_config = llm_config.model_copy(deep=True)
        eval_llm_provider = ZAVRageloLLMProvider(
            ChatClientFactory.create(eval_llm_client_config),
            max_tokens=eval_llm_client_config.model_configuration.max_tokens or 2048,
        )
        judge(
            queries,
            eval_llm_provider,
            reasoner_config,
            pairwise_evaluator_config if pairwise_clicked else custom_agent_eval_config,
        )
        eval_file_content.ragelo = RageloEvaluation(
            queries=queries,
            llm_config=llm_config,
            reasoner_config=reasoner_config_raw,
            custom_agent_eval_config=custom_agent_eval_config_raw,
            pairwise_config=pairwise_evaluator_config_raw,
            elo_ranker_config=agent_ranker_config_raw,
            elo_rankings=(
                rank_agents(queries, agent_ranker_config)
                if pairwise_clicked
                else ragelo.elo_rankings if ragelo else None
            ),
        )
        store_eval_file_content(eval_file_content)

    if eval_file_content.ragelo:
        if eval_file_content.ragelo.elo_rankings:
            st.subheader("Elo Rankings")
            elo_rankings_sorted = sorted(
                (
                    (agent_display_name(agent), elo)
                    for agent, elo in eval_file_content.ragelo.elo_rankings.items()
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            st.dataframe(pd.DataFrame(elo_rankings_sorted, columns=["Agent", "Elo"]))

            st.subheader("Pairwise Evaluations")
            pairwise_evaluator_name = str(pairwise_evaluator_config.evaluator_name)
            for q in eval_file_content.ragelo.queries:
                __render_query(q)
                for game in q.pairwise_games.values():
                    game_result = game.evaluations.get(pairwise_evaluator_name)
                    if not game_result or not game_result.winner:
                        st.write("No evaluation was generated for this game")
                        continue
                    verdict = "TIE" if game_result.winner == "C" else game_result.winner
                    with st.expander(
                        f"{agent_display_name(game.agent_a_answer.agent)}(A) VS "
                        f"{agent_display_name(game.agent_b_answer.agent)}(B) \n"
                        f"--- VERDICT: {verdict}"
                    ):
                        __render_message(
                            ChatMessageSender.BOT, game.agent_a_answer.rendered_text
                        )
                        __render_message(
                            ChatMessageSender.BOT, game.agent_b_answer.rendered_text
                        )
                        __render_cited_documents(
                            q, game.agent_a_answer, game.agent_b_answer
                        )
                        __render_verdict(
                            game_result.winner,
                            f"**A:** {game_result.answer_a_analysis}\n\n"
                            f"**B:** {game_result.answer_b_analysis}\n\n"
                            f"{game_result.comparison_reasoning}",
                        )
        else:
            st.write(
                "Elo rankings are not calculated yet. Click on `Evaluate Pairwise`"
            )

        st.subheader("Pointwise Evaluations")
        pointwise_evaluator_name = str(custom_agent_eval_config.evaluator_name)
        for q in eval_file_content.ragelo.queries:
            __render_query(q)
            answer_results = [
                (answer, answer.evaluations.get(pointwise_evaluator_name))
                for answer in q.answers.values()
            ]
            if not all(result for _, result in answer_results):
                st.write(f"No pointwise evaluation has been made for query: {q.query}")
                continue
            for answer, answer_result in answer_results:
                assert answer_result is not None
                with st.expander(agent_display_name(answer.agent)):
                    __render_message(ChatMessageSender.BOT, answer.rendered_text)
                    __render_cited_documents(q, answer)
                    __render_verdict(
                        str(answer_result.score), answer_result.reasoning or ""
                    )
