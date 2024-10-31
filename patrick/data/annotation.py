from abc import abstractmethod
from xml.etree.ElementTree import Element

from patrick.data.data_handler import DataHandler


class Annotation(DataHandler):
    def __init__(self, label: str):
        self._label = label

    @abstractmethod
    def rescale(self, w_ratio: str, h_ratio: str):
        pass

    @property
    def type(self) -> str:
        return type(self).__name__.lower()

    def to_dict(self):
        output = super().to_dict()
        return {"type": self.type, **output}


class Box(Annotation):
    def __init__(self, label: str, x: float, y: float, width: float, height: float):
        super().__init__(label)
        self._x = float(x)
        self._y = float(y)
        self._width = float(width)
        self._height = float(height)

    @property
    def xmin(self):
        return self._x

    @property
    def xmax(self):
        return self._x + self._width

    @property
    def ymin(self):
        return self._y

    @property
    def ymax(self):
        return self._y + self._height

    @staticmethod
    def _printable_fields():
        return ["label", "x", "y", "width", "height"]

    @classmethod
    def from_xml(cls, data_xml):
        return super().from_xml(data_xml)

    @classmethod
    def from_dict(cls, data_as_dict):
        return cls(**data_as_dict)

    def rescale(self, w_ratio, h_ratio):
        self._x = self._x * w_ratio
        self._y = self._y * h_ratio
        self._width = self._width * w_ratio
        self._height = self._height * w_ratio


class Polyline(Annotation):
    def __init__(
        self,
        label: str,
        point_list: list[tuple[float, float]],
    ):
        super().__init__(label)
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

    @classmethod
    def from_dict(cls, data_as_dict: dict):
        label = data_as_dict["label"]
        point_list = [tuple(point) for point in data_as_dict["point_list"]]
        return cls(label, point_list)

    def rescale(self, w_ratio: float, h_ratio: float):
        self._point_list = [(x * w_ratio, y * h_ratio) for x, y in self._point_list]


ANNOTATION_TYPE_DICT = {"polyline": Polyline}


def annotation_dict_factory(annotation_as_dict: dict) -> Annotation:
    annotation_type = annotation_as_dict["type"]
    annotation_class = ANNOTATION_TYPE_DICT[annotation_type]
    return annotation_class.from_dict(annotation_as_dict)


def annotation_xml_factory(annotation_xml: Element) -> Annotation:
    annotation_type = annotation_xml.tag
    annotation_class = ANNOTATION_TYPE_DICT[annotation_type]
    return annotation_class.from_xml(annotation_xml)


def parse_point_str(point_str: str) -> list[tuple[float, float]]:
    point_str_list = point_str.split(";")
    coord_str_list = [point_str.split(",") for point_str in point_str_list]
    return [(float(coord_str[0]), float(coord_str[1])) for coord_str in coord_str_list]
