from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from patrick.core import ComputingDevice, Movie, NeuralNet, NNModel
from patrick.interfaces import Builder, Repository

DATA_DIR_PATH = Path.home() / "data"
PATRICK_DIR_PATH = DATA_DIR_PATH / "pattern_discovery"


class LocalRepository(Repository):
    data_source = "local"
    name: str

    def __init__(self):
        self._directory_path = PATRICK_DIR_PATH / self.name


class LocalTorchNetRepository(LocalRepository):
    data_source = "local"
    name = "models"

    def __init__(self, device: ComputingDevice):
        super().__init__()
        self._device = device

    def read(self, content_path: str or Path) -> dict[str, Any]:
        full_content_path = self._directory_path / content_path
        with Path.open(full_content_path / "model_parameters.yaml") as f:
            net_parameters = yaml.safe_load(f)["net"]
        return {
            "weights": torch.load(
                full_content_path / "net.pth", map_location=self._concrete_device
            ),
            "net_parameters": net_parameters,
        }

    def write(self, content_path: str or Path, content: Any) -> None:
        full_content_path = self._directory_path / content_path
        torch.save(content, full_content_path)

    @property
    def _concrete_device(self) -> torch.DeviceObjType:
        torch_device_dict = {"cpu": torch.device("cpu"), "gpu": torch.device("cuda")}
        return torch_device_dict[self._device]


class LocalModelRepository(LocalRepository):
    data_source = "local"
    name = "models"
    _net_builder: Builder

    def read(self, content_path: str or Path) -> dict[str, dict or Any]:
        label_map = self._load_label_map(content_path)
        model_parameters = self._load_model_parameters(content_path)
        net = self._load_net(content_path)
        return {
            "label_map": label_map,
            "model_parameters": model_parameters,
            "net": net,
        }

    def write(self, content_path: str or Path, content: NNModel) -> None:
        pass

    def _load_label_map(self, content_path: str or Path) -> dict[str, int]:
        full_content_path = self._directory_path / content_path / "label_map.yaml"
        with Path.open(full_content_path) as f:
            return yaml.safe_load(f)

    def _load_model_parameters(self, content_path: str or Path) -> dict[str, dict]:
        full_content_path = (
            self._directory_path / content_path / "model_parameters.yaml"
        )
        with Path.open(full_content_path) as f:
            return yaml.safe_load(f)

    def _load_net(self, content_path: str or Path) -> NeuralNet:
        self._net_builder.build(model_name=content_path)


class LocalMovieRepository(LocalRepository):
    data_source = "local"
    name: str

    def __init__(self, name: str):
        self.name = name
        self._directory_path = PATRICK_DIR_PATH / self.name

    def read(self, content_path: str or Path) -> Movie:
        full_content_path = self._directory_path / content_path

        with Path.open(full_content_path) as f:
            movie_as_dict = json.load(f)

        return Movie.from_dict(movie_as_dict)

    def write(self, content_path: str or Path, content: Movie) -> None:
        full_content_path = self._directory_path / content_path
        with Path.open(full_content_path, "w") as f:
            json.dump(content, f, indent=2)
