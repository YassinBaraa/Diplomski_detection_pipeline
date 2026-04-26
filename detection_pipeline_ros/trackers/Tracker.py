from abc import ABC, abstractmethod

class Tracker(ABC):
    @abstractmethod
    def track(self, frame):
        pass
