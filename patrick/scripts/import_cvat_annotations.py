from argparse import ArgumentParser

from patrick import PATRICK_DIR_PATH
from patrick.storage.cvat import import_annotations

parser = ArgumentParser()
parser.add_argument(
    "--xml_file_name",
    help="CVAT XML annotation file name",
)

args = parser.parse_args()
xml_file_path = PATRICK_DIR_PATH / f"annotations/{args.xml_file_name}"
import_annotations(xml_file_path, sort_frames=True)
