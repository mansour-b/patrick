from xml.etree.ElementTree import Element

import numpy as np

from patrick import PATRICK_DIR_PATH
from patrick.data.annotation import Annotation, annotation_factory
from patrick.data.data_handler import DataHandler


class Image(DataHandler):
    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        annotations: list[Annotation] = None,
        image_array: np.array = None,
    ):
        if annotations is None:
            annotations = []
        self._name = name
        self._width = width
        self._height = height
        self._annotations = annotations
        self._image_array = image_array

    @staticmethod
    def _printable_fields():
        return ["name", "width", "height", "annotations"]

    @classmethod
    def from_xml(cls, data_xml: Element):
        attrib = data_xml.attrib
        return cls(
            name=attrib["name"].split(".")[0],
            width=attrib["width"],
            height=attrib["height"],
            annotations=[
                annotation_factory(annotation_xml) for annotation_xml in data_xml
            ],
        )

    def get_image_array(self, image_dir_name: str = None):
        if self._image_array is not None:
            return self._image_array

        file_path = PATRICK_DIR_PATH / f"input/{image_dir_name}/{self._name}.txt"
        return np.loadtxt(file_path)
