from abc import ABC, abstractmethod

from patrick import Annotation, Box, Frame, Keypoint


class Model(ABC):
    @abstractmethod
    def predict(frame: Frame) -> Annotation:
        pass


class BoxModel(Model):
    def predict(frame: Frame) -> Box:
        pass


class KeypointModel(Model):
    def predict(frame: Frame) -> Keypoint:
        pass
