from patrick.data.annotation import Box, Polyline
from patrick.data.image import Image


def serialise_image_list(image_list: list[Image]) -> list[dict]:
    return [image.to_dict() for image in image_list]


def deserialise_image_list(image_as_dict_list: list[dict]) -> list[Image]:
    return [Image.from_dict(image_as_dict) for image_as_dict in image_as_dict_list]


def polyline_to_box(polyline: Polyline) -> Box:
    points = polyline._point_list
    xmin = min(point[0] for point in points)
    xmax = max(point[0] for point in points)
    ymin = min(point[1] for point in points)
    ymax = max(point[1] for point in points)
    return Box(
        label=polyline._label, x=xmin, y=ymin, width=xmax - xmin, height=ymax - ymin
    )
