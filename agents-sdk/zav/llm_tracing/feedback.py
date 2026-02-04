from abc import ABC, abstractmethod
from typing import Optional, Union


class FeedbackService(ABC):
    @abstractmethod
    def __init__(self, vendor_configuration):
        raise NotImplementedError

    @abstractmethod
    async def add_score(
        self,
        trace_id: str,
        name: str,
        value: Union[str, float],
        comment: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def validate_trace_ownership(
        self,
        trace_id: str,
        user_id: str,
    ) -> bool:
        raise NotImplementedError
