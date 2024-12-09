from patrick.entities.annotation import Track
from patrick.entities.frame import Frame
from patrick.entities.metadata import Metadata


class Movie(Metadata):
    def __init__(self, name: str, frames: list[Frame], tracks: list[Track]):
        self.name = name
        self.frames = frames
        self.tracks = tracks
