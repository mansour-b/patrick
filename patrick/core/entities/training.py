from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    """Abstract class to represent a training algorithm.

    Concrete classes that inherit from it have to implement a fit() method,
    which will be used to tune the parameters of a neural network or the atoms
    of a convolutional dictionary.

    """

    @abstractmethod
    def fit(self, data: Any) -> None:
        """Fit the model on the data."""
