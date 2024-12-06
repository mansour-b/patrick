from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Repository(ABC):
    @abstractmethod
    def read(self, content_path: str or Path) -> Any:
        pass

    @abstractmethod
    def write(self, content_path, content) -> None:
        pass
