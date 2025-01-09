import numpy as np

from patrick.cd_training.base import GradientDescentCSC
from patrick.core import ConvDict


class NumpyGradientDescentCSC(GradientDescentCSC):
    def __call__(
        self, data: np.array, conv_dict: ConvDict, n_iterations: int
    ) -> np.array:
        num_channels = data.shape[0]
        data_shape = data.shape[1:]

        num_atoms = conv_dict.shape[0]
        atom_shape = conv_dict.shape[2:]

        activation_vector_shape = tuple(
            np.array(data_shape) - np.array(atom_shape) + 1
        )

        activation_vector = np.zeros(activation_vector_shape)
        auxiliary_activation_vector = np.zeros(activation_vector_shape)
        step_size = 1.0

        for _ in range(n_iterations):
            new_activation_vector = self.soft_thresholding(
                auxiliary_activation_vector
                - (1 / L)
                * conv_dict.transpose()
                * (np.conv(conv_dict, auxiliary_activation_vector) - data)
            )
            new_step_size = 0.5 * (1 + np.sqrt(1 + 4 * step_size**2))
            auxiliary_activation_vector = new_activation_vector + (
                step_size - 1
            ) / new_step_size * (new_activation_vector - activation_vector)

            activation_vector = new_activation_vector
            step_size = new_step_size
        return activation_vector

    def soft_thresholding(self):
        pass

    def compute_lipschitz_constant(self, conv_dict: ConvDict):
        pass
