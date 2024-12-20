from abc import ABC, abstractmethod

from patrick.core import Dataset, NeuralNet


class NNTrainer(ABC):
    def __init__(self, net: NeuralNet, dataset: Dataset):
        pass

    @abstractmethod
    def train(self) -> None:
        pass
