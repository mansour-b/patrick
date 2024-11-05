import json

import numpy as np
import tensorflow as tf
from tensorflow.train import BytesList, Example, Feature, Features, FloatList, Int64List

from patrick import PATRICK_DIR_PATH
from patrick.data.image import Image
from patrick.data.operations import deserialise_image_list


def make_tfrecords(experiment: str, image_width: int, image_height: int):

    image_list = load_image_list(experiment, image_width, image_height)

    output_file_path = PATRICK_DIR_PATH / f"tfrecords/{experiment}.tfrecord"

    with tf.io.TFRecordWriter(str(output_file_path)) as writer:
        for image in image_list:
            example = image_to_example(image, data_dir_name=experiment)
            writer.write(example.SerializeToString())


def cast_to_int(image_array: np.array, base: int = 8) -> np.array:
    min_val = image_array.min()
    max_val = image_array.max()
    max_range = max_val - min_val

    normalised_image = (image_array - min_val) / max_range

    int_dict = {
        8: {"max_value": 255, "type": np.uint8},
        16: {"max_value": 65535, "type": np.uint16},
    }
    max_int_val = int_dict[base]["max_value"]
    output_type = int_dict[base]["type"]

    return (max_int_val * normalised_image).astype(dtype=output_type)


def add_channels_dim(image_array: np.array) -> np.array:
    return np.expand_dims(image_array, axis=-1)


def image_to_example(image: Image, data_dir_name: str):

    int_image_array = cast_to_int(image.get_image_array(data_dir_name))
    int_image_array = add_channels_dim(int_image_array)
    image_bytes = tf.io.encode_png(int_image_array).numpy()

    normalised_box_coords = compute_normalised_box_coordinates(image)
    label_list = [box._label.encode("utf8") for box in image.get_boxes()]
    feature_dict = {
        "image/height": Feature(int64_list=Int64List(value=[image._height])),
        "image/width": Feature(int64_list=Int64List(value=[image._width])),
        "image/file_name": Feature(
            bytes_list=BytesList(value=[image._name.encode("utf8")])
        ),
        "image/encoded": Feature(bytes_list=BytesList(value=[image_bytes])),
        "image/format": Feature(bytes_list=BytesList(value=["png".encode("utf8")])),
        **{
            f"image/object/bbox/{k}": Feature(float_list=FloatList(value=v))
            for k, v in normalised_box_coords.items()
        },
        "image/object/class/object_type": Feature(
            bytes_list=BytesList(value=label_list)
        ),
    }

    example = Example(features=Features(feature=feature_dict))
    return example


def compute_normalised_box_coordinates(image: Image) -> dict[str, list[float]]:
    xmin_list = [box.xmin / image._width for box in image.get_boxes()]
    xmax_list = [box.xmax / image._width for box in image.get_boxes()]
    ymin_list = [box.ymin / image._height for box in image.get_boxes()]
    ymax_list = [box.ymax / image._height for box in image.get_boxes()]
    return {
        "xmin": xmin_list,
        "xmax": xmax_list,
        "ymin": ymin_list,
        "ymax": ymax_list,
    }


def load_image_list(experiment: str, image_width: int = None, image_height: int = None):
    file_path = PATRICK_DIR_PATH / f"annotations/{experiment}.json"

    with open(file_path) as f:
        image_list = deserialise_image_list(json.load(f))

    if image_width is None or image_height is None:
        return image_list

    for image in image_list:
        image.resize(image_width, image_height)
    return image_list
