from __future__ import annotations

from patrick.entities.annotation import Annotation, annotation_dict_factory
from patrick.entities.metadata import Metadata


class Frame(Metadata):

    def __init__(
        self, name: str, width: int, height: int, annotations: list[Annotation]
    ):
        self.name = name
        self.width = width
        self.height = height
        self.annotations = annotations

    @classmethod
    def printable_fields(cls):
        return ["name", "width", "height", "annotations"]

    @classmethod
    def from_dict(cls, data_as_dict: dict):
        return cls(
            name=data_as_dict["name"],
            width=data_as_dict["width"],
            height=data_as_dict["height"],
            annotations=[
                annotation_dict_factory(annotation_as_dict)
                for annotation_as_dict in data_as_dict["annotations"]
            ],
        )

    def to_dict(self):
        output = super().to_dict()
        output["annotations"] = [
            annotation.to_dict() for annotation in self.annotations
        ]
        return output

    def resize(self, target_width: int, target_height: int) -> None:
        w_ratio = target_width / self._width
        h_ratio = target_height / self._height

        self._width = target_width
        self._height = target_height

        for annotation in self._annotations:
            annotation.rescale(w_ratio, h_ratio)
