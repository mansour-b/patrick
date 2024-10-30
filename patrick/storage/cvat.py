from pathlib import Path
from xml.etree.ElementTree import parse

from patrick.data.image import Image


def import_annotations(file_path: Path, sort_frames: bool = True):
    """
    Open a CVAT XML annotation file,
    convert the annotations into patrick.data.image.Image format,
    and store them in a JSON file.
    """
    tree = parse(file_path)
    root = tree.getroot()
    image_list = [Image.from_xml(image_xml) for image_xml in root.findall("image")]
    if sort_frames:
        image_list.sort(key=lambda image: int(image._name.split("_")[-1]))
