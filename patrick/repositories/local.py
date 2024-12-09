from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from patrick import PATRICK_DIR_PATH, NNModel
from patrick.core import Movie
from patrick.core.entities.detection import NeuralNet
from patrick.interface.repository import Repository
from patrick.interfaces.builder import Builder


class LocalRepository(Repository):
    data_source = "local"
    name: str

    def __init__(self):
        self._directory_path = PATRICK_DIR_PATH / self.name


class LocalTorchNetRepository(LocalRepository):
    data_source = "local"
    name = "models"

    def read(self, content_path: str or Path) -> Any:
        full_content_path = self._directory_path / content_path
        return torch.load(full_content_path)

    def write(self, content_path: str or Path, content: Any) -> None:
        full_content_path = self._directory_path / content_path
        torch.save(content, full_content_path)


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
        self._net_builder.build(model_name=content_path.name)


class LocalMovieRepository(LocalRepository):
    data_source = "local"
    name: str

    def read(self, content_path: str or Path) -> Movie:
        full_content_path = self._directory_path / content_path

        with Path.open(full_content_path) as f:
            movie_as_dict = json.load(f)

        return Movie.from_dict(movie_as_dict)

    def write(self, content_path: str or Path, content: Movie) -> None:
        full_content_path = self._directory_path / content_path
        with Path.open(full_content_path, "w") as f:
            json.dump(content, f, indent=2)
