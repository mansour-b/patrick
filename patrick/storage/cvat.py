import json
from pathlib import Path
from xml.etree.ElementTree import parse

from patrick.data.image import Image
from patrick.data.operations import serialise_image_list


def import_annotations(xml_file_path: Path, sort_frames: bool = True):
    """
    Open a CVAT XML annotation file,
    convert the annotations into patrick.data.image.Image format,
    and store them in a JSON file.
    """
    tree = parse(xml_file_path)
    root = tree.getroot()

    image_list = [Image.from_xml(image_xml) for image_xml in root.findall("image")]
    if sort_frames:
        image_list.sort(key=lambda image: int(image._name.split("_")[-1]))

    output_path = xml_file_path.parent / f"{xml_file_path.stem}.json"
    with open(output_path, "w") as f:
        json.dump(serialise_image_list(image_list), f, indent=2)
