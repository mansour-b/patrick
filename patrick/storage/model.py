from abc import ABC, abstractmethod

import yaml

from patrick.entities.detection import Model, NNModel
from patrick.storage import PATRICK_DIR_PATH


class ModelRepository(ABC):
    @abstractmethod
    def load(self, model_name: str) -> Model:
        pass

    @abstractmethod
    def save(self, model: Model) -> None:
        pass


class NNModelRepository(ModelRepository):
    def load(self, model_name: str) -> NNModel:
        model_dir_path = PATRICK_DIR_PATH / f"models/{model_name}"

        net_path = model_dir_path / "net.pth"
        label_map_path = model_dir_path / "label_map.yaml"
        post_proc_param_path = model_dir_path / "post_processing_parameters.yaml"

        with open(label_map_path) as f:
            label_map = yaml.safe_load(f)
        with open(post_proc_param_path) as f:
            post_processing_parameters = yaml.safe_load(f)
