import numpy as np
import torch
from torch.utils.data import Dataset

from patrick.data.annotation import Box, Polyline
from patrick.data.image import Image
from patrick.data.operations import polyline_to_box


class BoxDataset(Dataset):
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


class KeypointDataset(Dataset):
    def __init__(
        self,
        image_list: list[Image],
        image_dir_name: str,
        label_map: dict[str, int],
        box_w_padding: float = 0.5,
        box_h_padding: float = 0.5,
    ):
        self._image_list = image_list
        self._image_dir_name = image_dir_name
        self._label_map = label_map
        self._w_padding = box_w_padding
        self.pox_h_padding = box_h_padding

    def __getitem__(self, index: int):
        image = self._image_list[index]

        image_array = self.load_image_array(image)
        image_tensor = torch.from_numpy(image_array)

        box_list = [
            polyline_to_box(
                polyline, w_padding=self._w_padding, h_padding=self.pox_h_padding
            )
            for polyline in image._annotations
        ]
        box_array = self.make_xyxy_box_array(box_list)
        label_array = self.make_label_array(box_list)
        area_array = self.make_area_array(box_list)
        keypoint_array = self.make_keypoint_array(image._annotations)

        target = {
            "image_id": index,
            "boxes": torch.as_tensor(box_array, dtype=torch.float32),
            "area": torch.as_tensor(area_array),
            "labels": torch.as_tensor(label_array),
            "keypoints": torch.as_tensor(keypoint_array, dtype=torch.float32),
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
    def make_keypoint_array(keypoint_list: list[Polyline]) -> np.array:
        return np.array(
            [
                [(x, y, 1.0) for x, y in polyline._point_list]
                for polyline in keypoint_list
            ]
        )

    @staticmethod
    def make_area_array(box_list: list[Box]) -> np.array:
        return np.array([box._width * box._height for box in box_list])
