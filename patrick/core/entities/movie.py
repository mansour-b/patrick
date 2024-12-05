from patrick.entities.annotation import Track
from patrick.entities.frame import Frame
from patrick.entities.metadata import Metadata


class Movie(Metadata):
    def __init__(self, frame_list: list[Frame], track_list: list[Track]):
        self.frame_list = frame_list
        self.track_list = track_list
