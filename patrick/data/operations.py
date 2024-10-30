from patrick.data.image import Image


def serialise_image_list(image_list: list[Image]) -> list[dict]:
    return [image.to_dict() for image in image_list]


def deserialise_image_list(image_as_dict_list: list[dict]) -> list[Image]:
    return [Image.from_dict(image_as_dict) for image_as_dict in image_as_dict_list]
