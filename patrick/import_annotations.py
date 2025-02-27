# %%
from __future__ import annotations

import json
from pathlib import Path
from xml.etree.ElementTree import Element, parse

import numpy as np

from patrick.core import Frame, Keypoint, Movie
from patrick.display import plot_frame


# %%
def parse_point_str(point_str: str) -> list[tuple[float, float]]:
    point_str_list = point_str.split(";")
    coord_str_list = [point_str.split(",") for point_str in point_str_list]
    return [
        (float(coord_str[0]), float(coord_str[1]))
        for coord_str in coord_str_list
    ]


def make_keypoint_from_xml(data_xml: Element):
    attrib = data_xml.attrib
    label = attrib["label"]
    point_list = parse_point_str(attrib["points"])
    return Keypoint(label, point_list, score=1.0)


def make_frame_from_xml(data_xml: Element):
    attrib = data_xml.attrib
    frame_name = int(attrib["name"].split(".")[0].split("_")[-1])
    return Frame(
        name=frame_name,
        width=int(attrib["width"]),
        height=int(attrib["height"]),
        annotations=[
            make_keypoint_from_xml(annotation_xml)
            for annotation_xml in data_xml
        ],
    )


# %%
PATRICK_DIR_PATH = Path.home() / "data" / "pattern_discovery"
file_path = PATRICK_DIR_PATH / "annotations/blob_i.xml"

# %%
tree = parse(file_path)
root = tree.getroot()


# %%
frames = [make_frame_from_xml(image_xml) for image_xml in root.findall("image")]
frames.sort(key=lambda frame: frame.name)

# %%
print(len(frames))
# %%
for frame in frames:
    frame.resize(512, 512)

# %%
print(frames[0])
# %%

for frame in frames:
    plot_frame(
        Frame(
            name=frame.name,
            width=frame.width,
            height=frame.height,
            annotations=frame.annotations,
            image_array=np.zeros((512, 512)),
        ),
    )

# %%
movie = Movie(name="blob_i/density_annotated", frames=frames, tracks=[])

# %%
movie_path = (
    PATRICK_DIR_PATH / "input" / "blob_i" / "density_annotated_movie.json"
)
with open(movie_path, "w") as f:
    json.dump(movie.to_dict(), f, indent=2)

# %%
