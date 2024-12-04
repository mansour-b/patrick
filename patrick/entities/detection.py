from abc import ABC, abstractmethod
from typing import Any

from patrick.entities.annotation import Annotation
from patrick.entities.array import Array
from patrick.entities.frame import Frame


class Model(ABC):
    @abstractmethod
    def predict(self, frame: Frame) -> Frame:
        pass


class NeuralNet(ABC):
    pass


class NNModel(Model):
    net: NeuralNet

    def predict(self, frame: Frame) -> Frame:
        input_array = self.pre_process(frame)
        predictions = self.net(input_array)
        annotations = self.post_process(predictions)
        return Frame(
            name=frame.name,
            width=frame.width,
            height=frame.height,
            annotations=annotations,
            image_array=frame.image_array,
        )

    @abstractmethod
    def pre_process(self, frame: Frame) -> Array:
        pass

    @abstractmethod
    def post_process(self, net_predictions: Any) -> list[Annotation]:
        pass


class CDModel(Model):
    pass
