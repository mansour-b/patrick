from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from patrick.core.entities.annotation import Annotation
from patrick.core.entities.array import Array
from patrick.core.entities.frame import Frame


class Model(ABC):
    @abstractmethod
    def predict(self, frame: Frame) -> Frame:
        pass


class NeuralNet:
    @abstractmethod
    def __call__(self, input_array: Array) -> Any:
        pass


class NNModel(Model):
    def __init__(
        self,
        net: NeuralNet,
        label_map: dict[str, int],
        model_parameters: dict,
    ):
        self.net = net
        self.label_map = label_map
        self.model_parameters = model_parameters

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
