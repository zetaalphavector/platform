from abc import ABC
from typing import ClassVar, Optional

from zav.agents_sdk.adapters.llm_models.chat_turn_request import InnerLoopContext
from zav.agents_sdk.domain.agent_dependency import DependencyGroup


class InterjectionSource(ABC):
    """Speaks to the agent from inside the model/tool loop. Override one or both
    seams; each returns the message text or None. ``before_model_call`` runs
    before every model call, with the transcript the model is about to receive.
    ``after_model_call`` runs once the model's answer (``context.response``) and
    the results of the tools it called are in the transcript."""

    source_name: ClassVar[str]
    # One line for admins choosing sources, e.g. in the agent editor.
    source_description: ClassVar[Optional[str]] = None
    # The tenant gate that hides this source until a tenant enables it.
    source_gate: ClassVar[Optional[str]] = None
    enabled: bool

    async def before_model_call(self, context: InnerLoopContext) -> Optional[str]:
        return None

    async def after_model_call(self, context: InnerLoopContext) -> Optional[str]:
        return None


class InterjectionSourceGroup(DependencyGroup[InterjectionSource]):
    __collects__ = InterjectionSource
