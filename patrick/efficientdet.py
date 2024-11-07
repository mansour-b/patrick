import numpy as np
import torch
from torch.utils.data import Dataset

from patrick.data.annotation import Box
from patrick.data.image import Image


class EfficientDetDataset(Dataset):
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

        box_list = image.get_boxes()
        box_array = self.make_xyxy_box_array(box_list)
        label_array = self.make_label_array(box_list)

        target = {
            "bboxes": torch.as_tensor(box_array, dtype=torch.float32),
            "labels": torch.as_tensor(label_array),
            "image_id": torch.tensor([index]),
            "img_size": (image._width, image._height),
            "img_scale": torch.tensor([1.0]),
        }

        return image_array, target, index

    def __len__(self):
        return len(self._image_list)

    def load_image_array(self, image: Image) -> np.array:
        image_array = image.get_image_array(self._image_dir_name).astype(
            np.float32, casting="same_kind"
        )
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    @staticmethod
    def box_to_xyxy_format(box: Box) -> list[float]:
        return [box.xmin, box.ymin, box.xmax, box.ymax]

    def make_xyxy_box_array(self, box_list: list[Box]) -> np.array:
        return np.array([self.box_to_xyxy_format(box) for box in box_list])

    def make_label_array(self, box_list: list[Box]) -> np.array:
        label_map = self._label_map
        return np.array([label_map[box._label] for box in box_list])
