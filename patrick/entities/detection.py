from abc import ABC, abstractmethod

from patrick.entities.annotation import Annotation, Box, Keypoint
from patrick.entities.frame import Frame


class Model(ABC):
    @abstractmethod
    def predict(self, frame: Frame) -> list[Annotation]:
        pass


class BoxModel(Model):
    def predict(self, frame: Frame) -> list[Box]:
        pass


class KeypointModel(Model):
    def predict(self, frame: Frame) -> list[Keypoint]:
        pass
