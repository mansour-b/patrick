from abc import abstractmethod
from typing import Any

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from patrick.core import ComputingDevice, NeuralNet
from patrick.interfaces.builder import Builder
from patrick.interfaces.repository import Repository


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


class TorchNetBuilder(NetBuilder):

    def _define_architecture(self, model_name: str, num_classes: int) -> NeuralNet:
        net = fasterrcnn_resnet50_fpn()
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return net

    def _load_weights(self, net: NeuralNet, weights: dict) -> None:
        net.load_state_dict(weights)

    def _migrate_on_computing_device(self, net: NeuralNet) -> None:
        torch_device_dict = {"cpu": torch.device("cpu"), "gpu": torch.device("cuda")}
        concrete_device = torch_device_dict[self._device]
        net.to(concrete_device)
