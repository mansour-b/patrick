import json
from pathlib import Path

from patrick.data.image import Image
from patrick.data.operations import deserialise_image_list, polyline_to_box


def load_images(
    annotation_file_path: Path, image_size: int, polyline_to_box_padding: float
) -> list[Image]:
    with open(annotation_file_path) as f:
        image_list = deserialise_image_list(json.load(f))

    for image in image_list:
        image.resize(image_size, image_size)

    image_list = [
        Image(
            name=image._name,
            width=image._width,
            height=image._height,
            annotations=[
                polyline_to_box(
                    polyline,
                    w_padding=polyline_to_box_padding,
                    h_padding=polyline_to_box_padding,
                )
                for polyline in image._annotations
            ],
        )
        for image in image_list
    ]
    return image_list
