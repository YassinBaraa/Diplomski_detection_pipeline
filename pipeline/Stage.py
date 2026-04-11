from abc import ABC, abstractmethod

class Stage(ABC):
    @abstractmethod
    def run(self, ctx):
        pass