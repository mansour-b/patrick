from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from patrick.core import ComputingDevice, Frame, Movie, NeuralNet, NNModel
from patrick.interfaces import Builder, Repository

DATA_DIR_PATH = Path.home() / "data"
PATRICK_DIR_PATH = DATA_DIR_PATH / "pattern_discovery"


class LocalRepository(Repository):
    data_source = "local"
    name: str

    def __init__(self):
        self._directory_path = PATRICK_DIR_PATH / self.name
        self._directory_path.mkdir(parents=True, exist_ok=True)


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
                full_content_path / "net.pth",
                map_location=self._concrete_device,
            ),
            "net_parameters": net_parameters,
        }

    def write(self, content_path: str or Path, content: Any) -> None:
        full_content_path = self._directory_path / content_path
        torch.save(content, full_content_path)

    @property
    def _concrete_device(self) -> torch.DeviceObjType:
        torch_device_dict = {
            "cpu": torch.device("cpu"),
            "gpu": torch.device("cuda"),
        }
        return torch_device_dict[self._device]


class LocalModelRepository(LocalRepository):
    data_source = "local"
    name = "models"
    _net_builder: Builder

    def read(self, content_path: str or Path) -> dict[str, dict or Any]:
        label_map = self._load_label_map(content_path)
        model_parameters = self._load_model_parameters(content_path)
        net = self._load_net(content_path)
        net.eval()
        return {
            "label_map": label_map,
            "model_parameters": model_parameters,
            "net": net,
        }

    def write(self, content_path: str or Path, content: NNModel) -> None:
        pass

    def _load_label_map(self, content_path: str or Path) -> dict[str, int]:
        full_content_path = (
            self._directory_path / content_path / "label_map.yaml"
        )
        with Path.open(full_content_path) as f:
            return yaml.safe_load(f)

    def _load_model_parameters(
        self, content_path: str or Path
    ) -> dict[str, dict]:
        full_content_path = (
            self._directory_path / content_path / "model_parameters.yaml"
        )
        with Path.open(full_content_path) as f:
            return yaml.safe_load(f)

    def _load_net(self, content_path: str or Path) -> NeuralNet:
        return self._net_builder.build(model_name=content_path)


class LocalMovieRepository(LocalRepository):
    data_source = "local"
    name: str

    def __init__(self, name: str):
        self.name = name
        super().__init__()

    def read(self, content_path: str or Path) -> Movie:
        experiment, field = self._parse_movie_name(movie_name=str(content_path))
        full_content_path = (
            self._directory_path / experiment / f"{field}_movie.json"
        )

        with Path.open(full_content_path) as f:
            movie = Movie.from_dict(json.load(f))
        self._load_image_arrays(movie)

        return movie

    def write(self, content_path: str or Path, content: Movie) -> None:
        experiment, field = self._parse_movie_name(movie_name=str(content_path))
        full_content_path = (
            self._directory_path / experiment / f"{field}_movie.json"
        )
        full_content_path.parent.mkdir(exist_ok=True)

        with Path.open(full_content_path, "w") as f:
            json.dump(content.to_dict(), f, indent=2)

    @staticmethod
    def _parse_movie_name(movie_name: str) -> tuple[str, str]:
        experiment = "_".join(movie_name.split("_")[:-1])
        field = movie_name.split("_")[-1]
        return experiment, field

    def _load_image_array(self, movie: Movie, frame: Frame) -> None:
        experiment, field = self._parse_movie_name(movie_name=movie.name)
        frame_id = int(frame.name)
        image_array_path = (
            self._directory_path / f"{experiment}/{field}_frame_{frame_id}.txt"
        )
        frame.image_array = np.loadtxt(image_array_path)

    def _load_image_arrays(self, movie: Movie) -> None:
        for frame in movie.frames:
            self._load_image_array(movie, frame)
