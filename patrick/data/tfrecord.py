from pathlib import Path

import tensorflow as tf

from patrick.data.image import Image
from patrick.efficientdet.dataset import tfrecord_util as tfru


def image_to_example(image: Image, data_dir_path: Path):
    image_path = data_dir_path / image.name

    image_str = tf.io.read_file(image_path)

    normalised_box_coords = compute_normalised_box_coordinates(image)
    label_list = [box._label for box in image.get_boxes()]
    feature_dict = {
        "image/height": tfru.int64_feature(image.height),
        "image/width": tfru.int64_feature(image.width),
        "image/file_name": tfru.bytes_feature(image.name.encode("utf8")),
        "image/raw": tfru.bytes_feature(image_str),
        "image/format": tfru.bytes_feature("png".encode("utf8")),
        **{
            f"image/object/bbox/{k}": tfru.float_list_feature(v)
            for k, v in normalised_box_coords.items()
        },
        "image/object/class/object_type": tfru.bytes_list_feature(label_list),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def compute_normalised_box_coordinates(image: Image) -> dict[str, list[float]]:
    xmin_list = [box.xmin / image.width for box in image.get_boxes()]
    xmax_list = [box.xmax / image.width for box in image.get_boxes()]
    ymin_list = [box.ymin / image.height for box in image.get_boxes()]
    ymax_list = [box.ymax / image.height for box in image.get_boxes()]
    return {
        "xmin": xmin_list,
        "xmax": xmax_list,
        "ymin": ymin_list,
        "ymax": ymax_list,
    }
