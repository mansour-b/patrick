from __future__ import annotations

from io import BytesIO
from pathlib import Path

from osfclient import OSF

from patrick.interfaces import Repository

PATRICK_OSF_PROJECT_ID = "jtp4z"


class OSFRepository(Repository):
    data_source = "osf"

    def __init__(self, name: str):
        super().__init(name)
        osf = OSF()
        project = osf.project(PATRICK_OSF_PROJECT_ID)
        storage = project.storage("osfstorage")


class OSFNNModelRepository(OSFRepository):
    def read(self, content_path: str or Path) -> dict[str, str or BytesIO]:
        pass

    def write(self, content_path: str or Path):
        raise NotImplementedError


class OSFMovieRepository(OSFRepository):
    def read(self, content_path: str or Path):
        pass

    def write(self, content_path: str or Path):
        raise NotImplementedError
