import numpy as np
import torch
from torch.utils.data import Dataset

from patrick.data.annotation import Box
from patrick.data.image import Image


class TorchvisionDataset(Dataset):
    def __init__(
        self,
        image_list: list[Image],
        image_dir_name: str,
        label_map: dict[str, int],
    ):
        self._image_list = image_list
        self._image_dir_name = image_dir_name
        self._label_map = label_map

    def __getitem__(self, index: int):
        image = self._image_list[index]

        image_array = self.load_image_array(image)
        image_tensor = torch.from_numpy(image_array)

        box_list = image.get_boxes()
        box_array = self.make_xyxy_box_array(box_list)
        label_array = self.make_label_array(box_list)
        area_array = self.make_area_array(box_list)

        target = {
            "boxes": torch.as_tensor(box_array, dtype=torch.float32),
            "labels": torch.as_tensor(label_array),
            "image_id": index,
            "area": torch.as_tensor(area_array),
            "iscrowd": torch.zeros((len(box_list),), dtype=torch.int64),
        }

        return image_tensor, target

    def __len__(self):
        return len(self._image_list)

    def load_image_array(
        self, image: Image, channel_mode: str = "channels_first"
    ) -> np.array:
        image_array = image.get_image_array(self._image_dir_name).astype(
            np.float32, casting="same_kind"
        )
        channel_axis_dict = {"channels_first": 0, "channels_last": -1}
        channel_axis = channel_axis_dict[channel_mode]
        image_array = np.expand_dims(image_array, axis=channel_axis)
        return np.repeat(image_array, repeats=3, axis=channel_axis)

    @staticmethod
    def box_to_xyxy_format(box: Box) -> list[float]:
        return [box.xmin, box.ymin, box.xmax, box.ymax]

    def make_xyxy_box_array(self, box_list: list[Box]) -> np.array:
        return np.array([self.box_to_xyxy_format(box) for box in box_list])

    def make_label_array(self, box_list: list[Box]) -> np.array:
        label_map = self._label_map
        return np.array([label_map[box._label] for box in box_list])

    @staticmethod
    def make_area_array(box_list: list[Box]) -> np.array:
        return np.array([box._width * box._height for box in box_list])