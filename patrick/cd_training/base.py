from abc import ABC, abstractmethod
from typing import Any

from patrick.core import ActivationVector, Array, ConvDict, Trainer


class ConvDictTrainer(Trainer):
    """Parent class for Convolutional Dictionary Learning."""

    def __init__(self):
        self.conv_dict: ConvDict = 0


class UnrolledCDTrainer(ConvDictTrainer):
    pass


class ConvolutionalSparseCoder(ABC):
    @abstractmethod
    def __call__(self, data: Any, conv_dict: ConvDict) -> Array:
        pass


class DictionaryUpdater(ABC):
    @abstractmethod
    def __call__(
        self,
        data: Any,
        conv_dict: ConvDict,
        activation_vector: ActivationVector,
    ) -> ConvDict:
        pass


class AMCDTrainer(ConvDictTrainer):
    """Alternating Minimisation (AM) ConvDict Trainer."""

    def __init__(self):
        super().__init__()
        self.encoder: ConvolutionalSparseCoder = 0
        self.dict_updater: DictionaryUpdater = 0

    def compute_activation_vector(self, data: Any) -> ActivationVector:
        return self.encoder(data, self.conv_dict)

    def update_dictionary(
        self, data: Any, activation_vector: ActivationVector
    ) -> None:
        self.conv_dict = self.dict_updater(
            data, self.conv_dict, activation_vector
        )

    def fit(self, data: Any, num_epochs: int) -> None:
        for _ in range(num_epochs):
            activation_vector = self.compute_activation_vector(data)
            self.update_dictionary(data, activation_vector)
