from pathlib import Path

from patrick.entities.detection import BoxModel, KeypointModel, Model
from patrick.entities.metadata import Annotation, Box, Frame, Keypoint, Movie
from patrick.entities.tracking import SORTTracker, Tracker

DATA_DIR_PATH = Path.home() / "data"
PATRICK_DIR_PATH = DATA_DIR_PATH / "pattern_detection_tokam"
TOKAM_DIR_PATH = DATA_DIR_PATH / "tokam2d"

__all__ = [
    "Annotation",
    "Box",
    "BoxModel",
    "Frame",
    "Keypoint",
    "KeypointModel",
    "Model",
    "Movie",
    "SORTTracker",
    "Tracker",
]
