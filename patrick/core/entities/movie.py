from patrick.core.entities.annotation import Track
from patrick.core.entities.frame import Frame
from patrick.core.entities.metadata import Metadata


class Movie(Metadata):
    def __init__(self, name: str, frames: list[Frame], tracks: list[Track]):
        self.name = name
        self.frames = frames
        self.tracks = tracks
