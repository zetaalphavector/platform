from typing import Any, Dict, List

from ragelo import (
    Experiment,
    Query,
    get_agent_ranker,
    get_answer_evaluator,
    get_retrieval_evaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import (
    BaseAnswerEvaluatorConfig,
    BaseRetrievalEvaluatorConfig,
    EloAgentRankerConfig,
)

DEFAULT_REASONER_CONFIG: Dict[str, Any] = {"n_processes": 20}
DEFAULT_PAIRWISE_CONFIG: Dict[str, Any] = {
    "n_games_per_query": 15,
    "include_relevance_reasoning": True,
    "include_raw_documents": True,
    "factors": (
        "the correctness, completeness, and level of detail of the responses along with their strong grounding to the cited documents. Answers are well grounded if they can be fully supported and justified by the cited documents without containing information that is not present in the cited documents."  # noqa: E501
    ),
    "document_relevance_threshold": None,
    "n_processes": 20,
}
DEFAULT_CUSTOM_PROMPT_CONFIG: Dict[str, Any] = {
    "user_prompt": """You are an impartial judge for evaluating the quality of the responses provided by
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
on its relevance, accuracy, completeness, and precision.
4. **Assign a score**: Give the answer a single score between 0 and 2, where 2 is
the highest, and explain in your reasoning how the answer does on each aspect.

DOCUMENTS RETRIEVED:
{% for document in documents %}[{{ document.did }}] {{ document.text }}
{% endfor %}
User Query:
{{ query.query }}

Agent answer:
{{ answer.text }}
""",  # noqa: E501
    "n_processes": 20,
}
DEFAULT_ELO_RANKER_CONFIG: Dict[str, Any] = {"initial_score": 1000, "tournaments": 10}


def __in_memory_experiment(queries: List[Query]) -> Experiment:
    experiment = Experiment(experiment_name="agents-sdk-eval", save_on_disk=False)
    for query in queries:
        experiment.add_query(query, should_save=False)
    return experiment


def judge(
    queries: List[Query],
    llm_provider: BaseLLMProvider,
    retrieval_config: BaseRetrievalEvaluatorConfig,
    answer_config: BaseAnswerEvaluatorConfig,
) -> None:
    experiment = __in_memory_experiment(queries)
    get_retrieval_evaluator(
        config=retrieval_config, llm_provider=llm_provider
    ).evaluate_experiment(experiment)
    get_answer_evaluator(
        config=answer_config, llm_provider=llm_provider
    ).evaluate_experiment(experiment)


def rank_agents(queries: List[Query], config: EloAgentRankerConfig) -> Dict[str, float]:
    return get_agent_ranker(config=config).run(__in_memory_experiment(queries)).scores
