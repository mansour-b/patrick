from abc import ABC, abstractmethod


class Metadata(ABC):
    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls):
        pass


class Annotation(Metadata):
    pass


class Frame(Metadata):

    def __init__(
        self, name: str, width: int, height: int, annotation_list: list[Annotation]
    ):
        self.name = name
        self.width = width
        self.height = height
        self.annotation_list

    def make_empty_copy(self):
        return Frame(
            name=self.name, width=self.width, height=self.height, annotation_list=[]
        )


class Movie(Metadata):
    def __init__(self, frame_list: list[Frame]):
        self.frame_list = frame_list


class Box(Annotation):
    pass


class Keypoint(Annotation):
    pass


class Track(Annotation):
    pass
