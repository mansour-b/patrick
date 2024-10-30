from patrick.data.image import Image


def serialise_image_list(image_list: list[Image]) -> list[dict]:
    return [image.to_dict() for image in image_list]
