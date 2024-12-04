from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from patrick.cnn.cnn_factory import cnn_factory
from patrick.entities.detection import Model, NNModel
from patrick.storage import PATRICK_DIR_PATH


class ModelRepository(ABC):
    @abstractmethod
    def glob(self, pattern: str) -> list[str]:
        pass

    @abstractmethod
    def load(self, model_name: str) -> Model:
        pass

    @abstractmethod
    def save(self, model: Model) -> None:
        pass

    @staticmethod
    def get_model_type(self, model_name: str) -> str:
        return "_".join(model_name.split("_")[:-2])


class NNModelRepository(ModelRepository):
    def glob(self, pattern: str) -> list[str]:
        return sorted((PATRICK_DIR_PATH / "models").glob(pattern))

    def load(self, model_name: str) -> NNModel:
        model_dir_path = PATRICK_DIR_PATH / f"models/{model_name}"

        net_path = model_dir_path / "net.pth"
        label_map_path = model_dir_path / "label_map.yaml"
        post_proc_param_path = model_dir_path / "post_processing_parameters.yaml"

        with Path.open(label_map_path) as f:
            label_map = yaml.safe_load(f)
        with Path.open(post_proc_param_path) as f:
            post_processing_parameters = yaml.safe_load(f)

        model_type = self.get_model_type(model_name)
        return cnn_factory(model_type, net_path, label_map, post_processing_parameters)
