from abc import ABC, abstractmethod

from patrick import Annotation, Box, Frame, Keypoint


class Model(ABC):
    @abstractmethod
    def predict(self, frame: Frame) -> Annotation:
        pass


class BoxModel(Model):
    def predict(self, frame: Frame) -> Box:
        pass


class KeypointModel(Model):
    def predict(self, frame: Frame) -> Keypoint:
        pass
