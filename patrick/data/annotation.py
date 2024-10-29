from xml.etree.ElementTree import Element

import numpy as np

from patrick.data.data_handler import DataHandler


class Annotation(DataHandler):
    pass


class Polyline(Annotation):
    def __init__(
        self,
        label: str,
        point_list: dict[str, np.array],
    ):
        self._label = label
        self._point_list = point_list

    @staticmethod
    def _printable_fields():
        return ["label", "point_list"]

    @classmethod
    def from_xml(cls, data_xml: Element):
        attrib = data_xml.attrib
        label = attrib["label"]
        point_list = parse_point_str(attrib["points"])
        return cls(label, point_list)


def annotation_factory(annotation_xml: Element) -> Annotation:
    annotation_type_dict = {"polyline": Polyline}
    annotation_type = annotation_xml.tag
    annotation_class = annotation_type_dict[annotation_type]
    return annotation_class.from_xml(annotation_xml)


def parse_point_str(point_str: str) -> list[np.array]:
    point_str_list = point_str.split(";")
    coord_str_list = [point_str.split(",") for point_str in point_str_list]
    return [np.array(coord_str).astype(float) for coord_str in coord_str_list]
