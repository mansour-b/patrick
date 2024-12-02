from abc import abstractmethod

from typing_extensions import Self

from patrick.entities.metadata import Metadata


class Annotation(Metadata):
    def __init__(self, label: str):
        self.label = label

    @abstractmethod
    def rescale(self, w_ratio: float, h_ratio: float):
        pass

    @property
    def type(self) -> str:
        return type(self).__name__.lower()


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
        super().__init__(label)
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
        self.score = float(score)

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

    @staticmethod
    def printable_fields() -> list[str]:
        """List of the relevant fields to serialise the object."""
        return ["type", "label", "x", "y", "width", "height", "score"]

    @classmethod
    def from_dict(cls, data_as_dict: dict) -> Self:
        """Make object from a dictionary."""
        init_params = {
            k: data_as_dict[k] for k in ["label", "x", "y", "width", "height", "score"]
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
    pass


class Track(Annotation):
    def __init__(self, track_id: int, box_list: list[Box]):
        self.track_id = track_id
        self.box_list = track_id
