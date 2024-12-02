from patrick.entities.annotation import Annotation
from patrick.entities.metadata import Metadata


class Frame(Metadata):

    def __init__(
        self, name: str, width: int, height: int, annotation_list: list[Annotation]
    ):
        self.name = name
        self.width = width
        self.height = height
        self.annotation_list = annotation_list
