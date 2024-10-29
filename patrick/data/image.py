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
        self._width = int(width)
        self._height = int(height)
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

    def resize(self, target_width: int, target_height: int):
        w_ratio = target_width / self._width
        h_ratio = target_height / self._height

        self._width = target_width
        self._height = target_height

        for annotation in self._annotations:
            annotation.rescale(w_ratio, h_ratio)

    def get_image_array(self, image_dir_name: str = None):
        if self._image_array is not None:
            return self._image_array

        file_path = PATRICK_DIR_PATH / f"input/{image_dir_name}/{self._name}.txt"
        image_array = np.loadtxt(file_path)

        array_shape = image_array.shape
        expected_shape = (self._width, self._height)

        if array_shape[::-1] != expected_shape:
            print(
                f"Warning: The image objects expects an array of shape "
                f"{expected_shape}. "
                f"The returned array has a shape {array_shape[::-1]}."
            )

        return image_array
