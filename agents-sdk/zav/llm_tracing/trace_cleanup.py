from abc import ABC, abstractmethod


class TraceCleanupService(ABC):
    @abstractmethod
    def __init__(self, vendor_configuration):
        raise NotImplementedError

    @abstractmethod
    async def delete_session_traces(self, session_id: str) -> None:
        raise NotImplementedError
