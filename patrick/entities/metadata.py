from abc import ABC


class Metadata(ABC):
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
