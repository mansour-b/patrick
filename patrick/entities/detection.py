from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from patrick.entities.annotation import Annotation
from patrick.entities.array import Array
from patrick.entities.frame import Frame


class Model(ABC):
    @abstractmethod
    def predict(self, frame: Frame) -> Frame:
        pass


class NNModel(Model):

    def __init__(
        self,
        net_path: Path,
        label_map: dict[str, int],
        post_processing_parameters: dict,
    ):
        self.net = self.get_net(net_path)
        self.label_map = label_map
        self.post_processing_parameters = post_processing_parameters

    @abstractmethod
    def get_net(self, net_path: Path) -> Callable[[Array], Any]:
        pass

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
