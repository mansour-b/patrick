import tensorflow as tf

from patrick.data.image import Image
from patrick.efficientdet.dataset import tfrecord_util as tfru


def image_to_example(image: Image, data_dir_name: str):

    image_bytes = image.get_image_array(data_dir_name).tobytes()

    normalised_box_coords = compute_normalised_box_coordinates(image)
    label_list = [box._label.encode("utf8") for box in image.get_boxes()]
    feature_dict = {
        "image/height": tfru.int64_feature(image._height),
        "image/width": tfru.int64_feature(image._width),
        "image/file_name": tfru.bytes_feature(image._name.encode("utf8")),
        "image/raw": tfru.bytes_feature(image_bytes),
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
