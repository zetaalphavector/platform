from abc import ABC, abstractmethod


class AbstractEventPublisher(ABC):
    @abstractmethod
    def publish_event(self, event):
        raise NotImplementedError
