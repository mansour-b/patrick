from patrick.data.annotation import Box, Polyline
from patrick.data.image import Image


def serialise_image_list(image_list: list[Image]) -> list[dict]:
    return [image.to_dict() for image in image_list]


def deserialise_image_list(image_as_dict_list: list[dict]) -> list[Image]:
    return [Image.from_dict(image_as_dict) for image_as_dict in image_as_dict_list]


def polyline_to_box(polyline: Polyline, w_padding: float, h_padding: float) -> Box:
    xmin, xmax, ymin, ymax = get_bounding_box(polyline._point_list)

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    output_width = bbox_width * (1 + w_padding)
    output_height = bbox_height * (1 + h_padding)

    x = xmin - bbox_width * w_padding / 2
    y = ymin - bbox_height * h_padding / 2

    return Box(
        label=polyline._label, x=x, y=y, width=output_width, height=output_height
    )


def get_bounding_box(
    point_list: list[tuple[float, float]]
) -> tuple[float, float, float, float]:

    xmin = min(point[0] for point in point_list)
    xmax = max(point[0] for point in point_list)
    ymin = min(point[1] for point in point_list)
    ymax = max(point[1] for point in point_list)

    return xmin, xmax, ymin, ymax
