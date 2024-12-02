from abc import abstractmethod

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
    pass


class Keypoint(Annotation):
    pass


class Track(Annotation):
    def __init__(self, track_id: int, box_list: list[Box]):
        self.track_id = track_id
        self.box_list = track_id
