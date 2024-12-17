from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from osfclient import OSF

from patrick.interfaces import Repository

PATRICK_OSF_PROJECT_ID = "jtp4z"

osf = OSF()
project = osf.project(PATRICK_OSF_PROJECT_ID)
storage = project.storage("osfstorage")

STORAGE_DICT = {file.path[1:]: file for file in storage.files}


class OSFRepository(Repository):
    data_source = "osf"

    def __init__(self, name: str):
        super().__init(name)
        self._path = Path(name)
        self._storage_dict = {
            k: v for k, v in STORAGE_DICT.items() if k.split("/")[0] == name
        }

    def write(self, content_path: str or Path, content: Any) -> None:
        msg = "OSFRepository is intended to be read-only."
        raise NotImplementedError(msg)


class OSFNNModelRepository(OSFRepository):
    def read(self, content_path: str or Path) -> dict[str, dict or BytesIO]:
        return {
            "label_map": self._load_yaml_file(content_path, "label_map.yaml"),
            "model_parameters": self._load_yaml_file(
                content_path, "model_parameters.yaml"
            ),
            "raw_net": self._load_raw_net(content_path),
        }

    def _load_yaml_file(
        self, content_path: str or Path, file_name: str
    ) -> dict[str, int or dict]:
        yaml_file = self._storage_dict[
            str(self._path / content_path / file_name)
        ]
        buffer = BytesIO()
        yaml_file.write_to(buffer)
        return yaml.safe_load(buffer.getvalue())

    def _load_raw_net(self, content_path: str or Path):
        net_file = self._storage_dict[
            str(self._path / content_path / "net.pth")
        ]
        buffer = BytesIO()
        net_file.write_to(buffer)
        return buffer.getvalue()


class OSFMovieRepository(OSFRepository):

    def __init__(self, name: str):
        self.name = name

        folder_name = {"input_movies": "input"}[name]
        self._path = Path(folder_name)
        self._storage_dict = {
            k: v
            for k, v in STORAGE_DICT.items()
            if k.split("/")[0] == folder_name
        }

    def read(self, content_path: str or Path):
        pass

    @staticmethod
    def _parse_movie_name(movie_name: str) -> tuple[str, str]:
        return tuple(movie_name.split("/"))
