from abc import ABC, abstractmethod


class Metadata(ABC):
    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls):
        pass


class Frame(Metadata):
    pass


class Movie(Metadata):
    pass


class Annotation(Metadata):
    pass


class Box(Annotation):
    pass


class Keypoint(Annotation):
    pass


class Track(Annotation):
    pass
