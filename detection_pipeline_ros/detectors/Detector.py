from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def predict(self, frame):
        pass
