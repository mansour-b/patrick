from abc import abstractmethod
from typing import Any

from patrick.core import NeuralNet
from patrick.interfaces.repository import Builder, Repository
from patrick.value_objects import ComputingDevice


class NetBuilder(Builder):
    def __init__(self, device: ComputingDevice, net_repository: Repository):
        self._net_repository = net_repository
        self._device = device

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
