import hashlib
import json
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from ragelo import (
    AgentAnswer,
    Document,
    Query,
    get_agent_ranker,
    get_answer_evaluator,
    get_llm_provider,
    get_retrieval_evaluator,
)
from ragelo.types.configurations import (
    CustomPromptAnswerEvaluatorConfig,
    EloAgentRankerConfig,
    PairwiseEvaluatorConfig,
    ReasonerEvaluatorConfig,
)
from zav.logging import logger

from zav.agents_sdk.cli.pages.ui_collect import (
    EvaluationFileContent,
    RageloEvaluation,
    RageloLLMConfig,
    get_eval_file_names,
    retrieve_eval_file_content,
    storage_path,
    store_eval_file_content,
)
from zav.agents_sdk.cli.ui_app import (
    ChatEntry,
    ChatMessageItem,
    EvaluatorItem,
    agent_display_name,
    render_chat_configuration_item,
    render_entry,
    retrieve_trace_file_content,
    start_new_conversation,
)
from zav.agents_sdk.domain.chat_message import ChatMessage, ChatMessageSender

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

        existing_docs = (
            qid_to_query[answer_trace.qid].retrieved_docs
            if answer_trace.qid in qid_to_query
            else []
        )
        existing_agent_answers = (
            qid_to_query[answer_trace.qid].answers
            if answer_trace.qid in qid_to_query
            else []
        )
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
                existing_doc = next(
                    (doc for doc in existing_docs if doc.did == ragelo_did),
                    None,
                )
                if not existing_doc:
                    qid_to_query[answer_trace.qid].add_retrieved_doc(
                        Document(
                            did=ragelo_did,
                            text=content,
                        )
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
                if not any(ragelo_did == doc.did for doc in existing_docs):
                    qid_to_query[answer_trace.qid].add_retrieved_doc(
                        Document(
                            did=ragelo_did,
                            text=evidence.text_extract,
                        )
                    )
        else:
            logger.exception(
                "No documents or evidences found in trace "
                f"{answer_trace.trace_file_name}"
            )

        if any(
            agent_answer.agent == ragelo_agent and agent_answer.text == ragelo_answer
            for agent_answer in existing_agent_answers
        ):
            continue
        qid_to_query[answer_trace.qid].add_agent_answer(
            AgentAnswer(
                agent=ragelo_agent,
                text=ragelo_answer,
            )
        )

    return list(qid_to_query.values())


def __parse_relevance_into_score(answer: str):
    if "very relevant" in answer.lower():
        return 2
    if "somewhat relevant" in answer.lower():
        return 1
    if "not relevant" in answer.lower():
        return 0
    else:
        raise ValueError(f"Could not parse relevance into score: {answer}")


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
    ragelo = eval_file_content.ragelo
else:
    ragelo = None


if ragelo:
    llm_config = RageloLLMConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "LLM Configuration",
                value=json.dumps(ragelo.llm_config.dict(), indent=2),
                disabled=True,
            )
        )
    )
    reasoner_config = ReasonerEvaluatorConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "Reasoner Evaluator Configuration",
                value=json.dumps(ragelo.reasoner_config.dict(), indent=2),
                disabled=True,
            )
        )
    )
    custom_agent_eval_config = CustomPromptAnswerEvaluatorConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "Custom Prompt Answer Evaluator Configuration",
                value=json.dumps(ragelo.custom_agent_eval_config.dict(), indent=2),
                disabled=True,
            )
        )
    )
    pairwise_evaluator_config = PairwiseEvaluatorConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "Pairwise Evaluator Configuration",
                value=json.dumps(ragelo.pairwise_config.dict(), indent=2),
                disabled=True,
            )
        )
    )
    agent_ranker_config = EloAgentRankerConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "Elo Ranker Configuration",
                value=json.dumps(ragelo.elo_ranker_config.dict(), indent=2),
                disabled=False,
            )
        )
    )

else:
    llm_config = RageloLLMConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "LLM Configuration",
                value=json.dumps(
                    {
                        "llm_provider": "openai",
                        "model_name": "gpt-4o",
                        "max_tokens": "4096",
                    },
                    indent=2,
                ),
            )
        )
    )
    reasoner_config = ReasonerEvaluatorConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "Reasoner Evaluator Configuration",
                value=json.dumps({"n_processes": 20}, indent=2),
            )
        )
    )
    pairwise_evaluator_config = PairwiseEvaluatorConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "Pairwise Evaluator Configuration",
                value=json.dumps(
                    {
                        "n_games_per_query": 15,
                        "has_citations": True,
                        "include_annotations": True,
                        "include_raw_documents": True,
                        "factors": (
                            "the correctness, completeness, and level of detail of the responses along with their strong grounding to the cited documents. Answers are well grounded if they can be fully supported and justified by the cited documents without containing information that is not present in the cited documents."  # noqa: E501
                        ),
                        "document_relevance_threshold": None,
                        "n_processes": 20,
                    },
                    indent=2,
                ),
            )
        )
    )
    custom_agent_eval_config = CustomPromptAnswerEvaluatorConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "Custom Prompt Answer Evaluator Configuration",
                value=json.dumps(
                    {
                        "prompt": """You are an impartial judge for evaluating the quality of the responses provided by
an AI assistant tasked to answer user's question about the AI domain.

You will be given the user's question and the answer produced by the assistant.
The agent's answer was generated based on a set of documents retrieved by a
search engine.
You will be provided with the a set of relevant documents that were retrieved by
the search engine.
Your task is to evaluate the quality of the answer based on the relevance, accuracy,
and completeness of the response.

## Rules for evaluating an answer:
- **Relevance**: Does the answer address the user's question?
- **Accuracy**: Is the answer factually correct, based on the documents provided?
- **Completeness**: Does the answer provide all the information needed to answer
the user's question?
- **Precision**: If the user's question is about specific terms, does the
answer provide the answer for those specific terms?

## Steps to evaluate an answer:
1. **Understand the user's intent**: Explain in your own words what the user's
intent is, given the question.
2. **Check if the answer is correct**: Think step-by-step whether the answer
correctly answers the user's question.
3. **Evaluate the quality of the answer**: Evaluate the quality of the answer based
on its relevance, accuracy, and completeness.
4. **Assign a score**: Produce a single line JSON object with the following keys,
each with a single score between 0 and 2, where 2 is the highest score on that aspect:
    - "relevance"
        - 0: The answer is not relevant to the user's question.
        - 1: The answer is partially relevant to the user's question.
        - 2: The answer is fully relevant to the user's question.
    - "accuracy"
        - 0: The answer is factually incorrect.
        - 1: The answer is partially correct.
        - 2: The answer is fully correct.
    - "completeness"
        - 0: The answer does not provide enough information to answer the user's question.
        - 1: The answer only answers some aspects of the user's question.
        - 2: The answer fully answers the user's question.
    - "precision"
        - 0: The answer does not mention the same terms as the user's question.
        - 1: The answer mentions a similar terms, but not the same as the user's question.
        - 2: The answer mentions the exact same terms as the user's question.

The last line of your answer must be a SINGLE LINE JSON object with the keys "relevance",
"accuracy", "completeness", and "precision", each with a single score between 0 and 2.

DOCUMENTS RETRIEVED:
{documents}

User Query:
{query}

Agent answer:
{answer}
""",  # noqa: E501
                        "scoring_keys": [
                            "relevance",
                            "accuracy",
                            "completeness",
                            "precision",
                        ],
                        "n_processes": 20,
                    },
                    indent=2,
                ),
            )
        )
    )
    agent_ranker_config = EloAgentRankerConfig.parse_obj(
        json.loads(
            st.sidebar.text_area(
                "Elo Ranker Configuration",
                value=json.dumps(
                    {"initial_score": 1000, "rounds": 500},
                    indent=2,
                ),
            )
        )
    )

if sel_existing_eval:  # noqa
    eval_file_content = retrieve_eval_file_content(sel_existing_eval)
    st.session_state.eval_file_name = f"{storage_path}/agent-evals/{sel_existing_eval}"
    st.session_state.eval_file_content = eval_file_content

    st.subheader("Selected agents")
    for conf in eval_file_content.agent_configurations:
        render_chat_configuration_item(conf, True)

    pairwise_clicked = st.button("Evaluate pairwise")
    pointwise_clicked = st.button("Evaluate pointwise")
    if pairwise_clicked or pointwise_clicked:
        queries = __construct_ragelo_queries(eval_file_content)
        eval_file_content.ragelo = RageloEvaluation(
            queries=queries,
            llm_config=llm_config,
            reasoner_config=reasoner_config,
            custom_agent_eval_config=custom_agent_eval_config,
            pairwise_config=pairwise_evaluator_config,
            elo_ranker_config=agent_ranker_config,
            elo_rankings=ragelo.elo_rankings if ragelo else None,
        )
        store_eval_file_content(eval_file_content)

        eval_llm_provider = get_llm_provider(
            llm_config.llm_provider,
            model_name=llm_config.model_name,
            max_tokens=llm_config.max_tokens,
        )
        retrieval_evaluator = get_retrieval_evaluator(
            config=reasoner_config,
            llm_provider=eval_llm_provider,
        )
        queries = retrieval_evaluator.batch_evaluate(queries)
        for q in queries:
            for d in q.retrieved_docs:
                if not d.evaluation or not d.evaluation.raw_answer:
                    raise Exception(f"No evaluation or raw answer for document {d}")
                if isinstance(d.evaluation.answer, str):
                    d.evaluation.answer = __parse_relevance_into_score(
                        d.evaluation.raw_answer
                    )
        eval_file_content.ragelo.queries = queries
        store_eval_file_content(eval_file_content)

        if pairwise_clicked:
            if len(eval_file_content.agent_configurations) < 2:
                st.error(
                    "For a pairwise evaluation, please add at least two agents to "
                    "the evaluator."
                )
            else:
                pairwise_evaluator = get_answer_evaluator(
                    config=pairwise_evaluator_config, llm_provider=eval_llm_provider
                )
                result_queries = pairwise_evaluator.batch_evaluate(queries)
                for result_q, q in zip(result_queries, queries):
                    q.pairwise_games = result_q.pairwise_games
        elif pointwise_clicked:
            pointwise_evaluator = get_answer_evaluator(
                "custom_prompt",
                llm_provider=eval_llm_provider,
                config=custom_agent_eval_config,
            )
            result_queries = pointwise_evaluator.batch_evaluate(queries)
            updated_queries = []
            for result_q, q in zip(result_queries, queries):
                q.answers = result_q.answers
                updated_queries.append(q)
            queries = updated_queries

        eval_file_content.ragelo.queries = queries
        store_eval_file_content(eval_file_content)

        if pairwise_clicked:
            elo_ranker = get_agent_ranker(
                "elo",
                verbose=agent_ranker_config.verbose,
                initial_score=agent_ranker_config.initial_score,
                rounds=agent_ranker_config.rounds,
            )

            agent_name_2_elo = elo_ranker.run(queries)
            eval_file_content.ragelo.elo_rankings = agent_name_2_elo
            store_eval_file_content(eval_file_content)

    if eval_file_content.ragelo:
        if eval_file_content.ragelo.elo_rankings:
            st.subheader("Elo Rankings")
            elo_rankings = {
                agent_display_name(agent): elo
                for agent, elo in eval_file_content.ragelo.elo_rankings.items()
            }
            elo_rankings_sorted = sorted(
                elo_rankings.items(), key=lambda x: x[1], reverse=True
            )
            elo_rankings_df = pd.DataFrame(
                elo_rankings_sorted, columns=["Agent", "Elo"]
            )
            st.dataframe(elo_rankings_df)

            win_lose_tie_ratios: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(lambda: 0)
            )
            ties: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(lambda: 0)
            )
            games_played: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(lambda: 0)
            )

            total_wins: Dict[str, int] = defaultdict(lambda: 0)
            total_games_played: Dict[str, int] = defaultdict(lambda: 0)

            for q in eval_file_content.ragelo.queries:
                for game in q.pairwise_games:
                    if not game.evaluation or not game.evaluation.answer:
                        raise Exception(f"No answer for game {game}")
                    agent_a = game.agent_a_answer.agent
                    agent_b = game.agent_b_answer.agent
                    answer = game.evaluation.answer
                    games_played[agent_a][agent_b] += 1
                    games_played[agent_b][agent_a] += 1
                    total_games_played[agent_a] += 1
                    total_games_played[agent_b] += 1
                    if answer == "A":
                        win_lose_tie_ratios[agent_a][agent_b] += 1
                        total_wins[agent_a] += 1
                    elif answer == "B":
                        win_lose_tie_ratios[agent_b][agent_a] += 1
                        total_wins[agent_b] += 1
                    elif answer == "C":  # a tie
                        ties[agent_a][agent_b] += 1
                        ties[agent_b][agent_a] += 1

            st.subheader("Pairwise Evaluations")
            for q in eval_file_content.ragelo.queries:
                start_new_conversation()
                render_entry(
                    entries=st.session_state.entries,
                    entry=ChatEntry(
                        chat_message_item=ChatMessageItem(
                            message=ChatMessage(
                                sender=ChatMessageSender.USER,
                                content=q.query,
                            )
                        )
                    ),
                    print_debug_logs=False,
                )
                for game in q.pairwise_games:
                    if (eval := game.evaluation) and eval.answer and eval.raw_answer:
                        with st.expander(
                            f"{agent_display_name(game.agent_a_answer.agent)}(A) VS "
                            f"{agent_display_name(game.agent_b_answer.agent)}(B) \n"
                            f"--- VERDICT: {eval.answer if eval.answer != 'C' else 'TIE'}"  # noqa: E501
                        ):
                            render_entry(
                                entries=st.session_state.entries,
                                entry=ChatEntry(
                                    chat_message_item=ChatMessageItem(
                                        message=ChatMessage(
                                            sender=ChatMessageSender.BOT,
                                            content=game.agent_a_answer.text,
                                        )
                                    )
                                ),
                            )
                            render_entry(
                                entries=st.session_state.entries,
                                entry=ChatEntry(
                                    chat_message_item=ChatMessageItem(
                                        message=ChatMessage(
                                            sender=ChatMessageSender.BOT,
                                            content=game.agent_b_answer.text,
                                        )
                                    )
                                ),
                            )
                            for doc in q.retrieved_docs:
                                if (
                                    doc.did in game.agent_a_answer.text
                                    or doc.did in game.agent_b_answer.text
                                ):
                                    st.markdown(f"###### [{doc.did}]\n {doc.text}")
                            render_entry(
                                entries=st.session_state.entries,
                                entry=ChatEntry(
                                    evaluator_item=EvaluatorItem(
                                        verdict=str(eval.answer),
                                        explanation=eval.raw_answer,
                                    ),
                                ),
                            )
                    else:
                        st.write("No evaluation was generated for this game")
        else:
            st.write(
                "Elo rankings are not calculated yet. Click on `Evaluate Pairwise`"
            )

        st.subheader("Pointwise Evaluations")
        for q in eval_file_content.ragelo.queries:
            start_new_conversation()
            render_entry(
                entries=st.session_state.entries,
                entry=ChatEntry(
                    chat_message_item=ChatMessageItem(
                        message=ChatMessage(
                            sender=ChatMessageSender.USER,
                            content=q.query,
                        )
                    )
                ),
                print_debug_logs=False,
            )
            if any(not a.evaluation for a in q.answers):
                st.write(f"No pointwise evaluation has been made for query: {q.query}")
                continue
            for answer in q.answers:
                if answer.evaluation:
                    with st.expander(agent_display_name(answer.agent)):
                        render_entry(
                            entries=st.session_state.entries,
                            entry=ChatEntry(
                                chat_message_item=ChatMessageItem(
                                    message=ChatMessage(
                                        sender=ChatMessageSender.BOT,
                                        content=answer.text,
                                    )
                                )
                            ),
                        )
                        for doc in q.retrieved_docs:
                            if doc.did in answer.text:
                                st.markdown(f"###### [{doc.did}]\n {doc.text}")
                        render_entry(
                            entries=st.session_state.entries,
                            entry=ChatEntry(
                                evaluator_item=EvaluatorItem(
                                    verdict=(
                                        f"```{json.dumps(answer.evaluation.answer)}```"
                                        if isinstance(answer.evaluation.answer, dict)
                                        else str(answer.evaluation.answer)
                                    ),
                                    explanation=answer.evaluation.raw_answer or "",
                                ),
                            ),
                        )
