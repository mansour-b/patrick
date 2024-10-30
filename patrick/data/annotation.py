from abc import abstractmethod
from xml.etree.ElementTree import Element

from patrick.data.data_handler import DataHandler


class Annotation(DataHandler):
    pass

    @abstractmethod
    def rescale(self, w_ratio: str, h_ratio: str):
        pass

    @property
    def type(self) -> str:
        return type(self).__name__.lower()


class Polyline(Annotation):
    def __init__(
        self,
        label: str,
        point_list: list[tuple[float, float]],
    ):
        self._label = label
        self._point_list = [(float(coord[0]), float(coord[1])) for coord in point_list]

    @staticmethod
    def _printable_fields():
        return ["label", "point_list"]

    @classmethod
    def from_xml(cls, data_xml: Element):
        attrib = data_xml.attrib
        label = attrib["label"]
        point_list = parse_point_str(attrib["points"])
        return cls(label, point_list)

    def rescale(self, w_ratio: float, h_ratio: float):
        self._point_list = [(x * w_ratio, y * h_ratio) for x, y in self._point_list]


def annotation_factory(annotation_xml: Element) -> Annotation:
    annotation_type_dict = {"polyline": Polyline}
    annotation_type = annotation_xml.tag
    annotation_class = annotation_type_dict[annotation_type]
    return annotation_class.from_xml(annotation_xml)


def parse_point_str(point_str: str) -> list[tuple[float, float]]:
    point_str_list = point_str.split(";")
    coord_str_list = [point_str.split(",") for point_str in point_str_list]
    return [(float(coord_str[0]), float(coord_str[1])) for coord_str in coord_str_list]
