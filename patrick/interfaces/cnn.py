from abc import ABC, abstractmethod
from typing import Any

from patrick.entities import NeuralNet
from patrick.value_objects import ComputingDevice, DataSource, Framework


class NetRepository(ABC):
    def __init__(self, data_source: DataSource, framework: Framework):

        self._data_source = data_source
        self._framework = framework

    @abstractmethod
    def read(model_name: str) -> Any:
        pass


class NetBuilder(ABC):
    def __init__(
        self, data_source: DataSource, framework: Framework, device: ComputingDevice
    ):
        self._net_repository = NetRepository(
            data_source=data_source, framework=framework
        )
        self._device = device

    @abstractmethod
    def build(self, model_name: str) -> NeuralNet:
        net = self._define_architecture(model_name)
        weights = self._net_repository.read(model_name)
        self._load_weights(net, weights)
        self._migrate_on_computing_device(net)
        return net

    @abstractmethod
    def _define_architecture(self, model_name: str) -> NeuralNet:
        pass

    @abstractmethod
    def _load_weights(self, net: NeuralNet, weights: Any) -> None:
        pass

    @abstractmethod
    def _migrate_on_computing_device(self, net: NeuralNet) -> None:
        pass
