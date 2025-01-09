import torch

from patrick.cd_training.base import GradientDescentCSC
from patrick.core import ConvDict


class TorchGradientDescentCSC(GradientDescentCSC):
    def __call__(self, data: torch.Tensor, conv_dict: ConvDict) -> torch.Tensor:
        pass
