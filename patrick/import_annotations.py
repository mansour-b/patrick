# %%
from pathlib import Path
from xml.etree.ElementTree import parse

from patrick.core import Frame
from patrick.display import plot_frame


# %%
def make_frame_from_xml(element):
    pass


# %%
PATRICK_DIR_PATH = Path.home() / "data" / "pattern_discovery"
file_path = PATRICK_DIR_PATH / "annotations/blob_i.xml"

# %%
tree = parse(file_path)
root = tree.getroot()
