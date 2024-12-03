from abc import abstractmethod

from typing_extensions import Self

from patrick.entities.metadata import Metadata


class Annotation(Metadata):
    def __init__(self, label: str, score: float):
        self.label = label
        self.score = score

    @abstractmethod
    def rescale(self, w_ratio: float, h_ratio: float):
        pass

    @property
    def type(self) -> str:
        return type(self).__name__.lower()

    @classmethod
    def printable_fields(cls) -> list[str]:
        return ["type", "label", "score"]


class Box(Annotation):
    """Class to represent bounding boxes."""

    def __init__(
        self, label: str, x: float, y: float, width: float, height: float, score: float
    ):
        """Initialise the bounding box.

        Args:
            label (str): Name of the object localised by the box (e.g., cat).
            x (float): X-position of the top-left corner.
            y (float): Y-position of the top-left corner.
            width (float): Width of the bounding box.
            height (float): Height of the bounding box.
            score (float): Confidence score of the detection (in [0, 1]).

        """
        super().__init__(label, score)
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)

    @property
    def xmin(self) -> float:
        """Xmin for XYXY format."""
        return self.x

    @property
    def xmax(self) -> float:
        """Xmax for XYXY format."""
        return self.x + self.width

    @property
    def ymin(self) -> float:
        """Ymin for XYXY format."""
        return self.y

    @property
    def ymax(self) -> float:
        """Ymax for XYXY format."""
        return self.y + self.height

    @classmethod
    def printable_fields(cls) -> list[str]:
        """List of the relevant fields to serialise the object."""
        output = super().printable_fields()
        return [*output, "x", "y", "width", "height"]

    @classmethod
    def from_dict(cls, data_as_dict: dict) -> Self:
        """Make object from a dictionary."""
        init_params = {
            k: data_as_dict[k] for k in cls.printable_fields() if k != "type"
        }
        return cls(**init_params)

    def rescale(self, w_ratio: float, h_ratio: float) -> None:
        """Rescale object.

        Args:
            w_ratio (float): Width ratio.
            h_ratio (float): Height ratio.

        """
        self.x = self.x * w_ratio
        self.y = self.y * h_ratio
        self.width = self.width * w_ratio
        self.height = self.height * w_ratio


class Keypoint(Annotation):
    """Class to represent keypoint annotations (e.g., for pose estimation)."""

    def __init__(
        self,
        label: str,
        point_list: list[tuple[float, float]],
        score: float,
    ):
        """Initialise the keypoint object.

        Args:
            label (str): Name of the object localised by the keypoint object.
            point_list (list): List of points for the keypoint annotation in
                the format [(x0, y0), (x1, y1), ...].
            score (float): Confidence of the detection.

        """
        super().__init__(label, score)
        self.point_list = [(float(coord[0]), float(coord[1])) for coord in point_list]

    @classmethod
    def printable_fields(cls) -> list[str]:
        """List of the relevant fields to serialise the object."""
        output = super().printable_fields()
        return [*output, "label", "point_list", "score"]

    @classmethod
    def from_dict(cls, data_as_dict: dict) -> Self:
        """Make object from a dictionary."""
        label = data_as_dict["label"]
        point_list = [tuple(point) for point in data_as_dict["point_list"]]
        score = data_as_dict["score"]
        return cls(label, point_list, score)

    def rescale(self, w_ratio: float, h_ratio: float) -> None:
        """Rescale object.

        Args:
            w_ratio (float): Width ratio.
            h_ratio (float): Height ratio.

        """
        self.point_list = [(x * w_ratio, y * h_ratio) for x, y in self.point_list]


class Track(Annotation):
    def __init__(self, track_id: int, box_list: list[Box]):
        self.track_id = track_id
        self.box_list = box_list
