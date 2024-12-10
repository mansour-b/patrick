from __future__ import annotations

from typing import Any

from patrick.core import CDModel, Model, NNModel
from patrick.interfaces.builder import Builder
from patrick.interfaces.repository import Repository


def parse_model_name(model_name: str) -> dict[str, Any]:
    model_architecture = "_".join(model_name.split("_")[:-2])
    model_date = "_".join(model_name.split("_")[-2:])
    model_type = "cnn" if "cnn" in model_architecture else "cdl"
    return {"architecture": model_architecture, "date": model_date, "type": model_type}


class ModelBuilder(Builder):

    def __init__(self, model_name: str, model_repository: Repository):
        self._model_name = model_name
        self._model_type = parse_model_name(model_name)["type"]
        self._model_repository = model_repository
        self._concrete_model_class = {"cnn": NNModel, "cdl": CDModel}[self._model_type]

    def build(self) -> Model:
        model_as_dict = self._model_repository.read(self._model_name)
        return self._concrete_model_class(**model_as_dict)
