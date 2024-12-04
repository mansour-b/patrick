from abc import ABC, abstractmethod

from patrick.entities.frame import Frame


class Model(ABC):
    @abstractmethod
    def predict(self, frame: Frame) -> Frame:
        pass


class NNModel(Model):
    pass


class CDModel(Model):
    pass
