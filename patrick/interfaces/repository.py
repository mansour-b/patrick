from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from patrick.core import DataSource


class Repository(ABC):
    data_source: DataSource
    name: str

    @abstractmethod
    def read(self, content_path: str or Path) -> Any:
        pass

    @abstractmethod
    def write(self, content_path, content) -> None:
        pass
