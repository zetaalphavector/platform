from hashlib import sha1
from typing import Any, Dict, List, Optional

from ragelo import Query
from ragelo.types.configurations import (
    CustomPromptAnswerEvaluatorConfig,
    EloAgentRankerConfig,
    PairwiseEvaluatorConfig,
    ReasonerEvaluatorConfig,
)
from zav.message_bus import MessageBus
from zav.pydantic_compat import PYDANTIC_V2, BaseModel

from zav.agents_sdk import AgentSetup, ChatMessage, ConversationContext


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

    @classmethod
    def from_configuration(
        cls, chat_configuration_item: ChatConfigurationItem
    ) -> "ChatEntry":
        return (cls.model_construct if PYDANTIC_V2 else cls)(
            chat_configuration_item=chat_configuration_item
        )

    @classmethod
    def from_message(cls, chat_message_item: ChatMessageItem) -> "ChatEntry":
        return (cls.model_construct if PYDANTIC_V2 else cls)(
            chat_message_item=chat_message_item
        )

    @classmethod
    def from_evaluator(cls, evaluator_item: EvaluatorItem) -> "ChatEntry":
        return (cls.model_construct if PYDANTIC_V2 else cls)(
            evaluator_item=evaluator_item
        )


class ComputeChatMessageItem(BaseModel):
    message_bus: MessageBus
    chat_configuration_item: ChatConfigurationItem

    class Config:
        arbitrary_types_allowed = True


class TraceFileContent(BaseModel):
    entries: List[ChatEntry]

    @classmethod
    def from_entries(cls, entries: List[ChatEntry]) -> "TraceFileContent":
        return (cls.model_construct if PYDANTIC_V2 else cls)(entries=entries)


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
