from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision.ops import nms

from patrick.data.annotation import Box
from patrick.data.image import Image

if TYPE_CHECKING:
    from pathlib import Path


def load_image_array(
    file_path: str or Path,
    device: torch.device,
    channel_mode: str = "channels_first",
) -> np.array:
    image_array = np.loadtxt(file_path).astype(np.float32, casting="same_kind")
    channel_axis_dict = {"channels_first": 0, "channels_last": -1}
    channel_axis = channel_axis_dict[channel_mode]
    image_array = np.expand_dims(image_array, axis=channel_axis)
    image_array = np.repeat(image_array, repeats=3, axis=channel_axis)

    image_array = torch.as_tensor(
        [
            load_image_array(file_path),
        ]
    )
    image_array.to(device)
    return image_array


def xyxy_to_xywh(
    xmin: float, ymin: float, xmax: float, ymax: float
) -> tuple[float, float, float, float]:
    x = xmin
    y = ymin
    width = xmax - xmin
    height = ymax - ymin
    return x, y, width, height


def make_str_label(label: int, label_map: dict[str, int]) -> str:
    reciprocal_label_map = {v: k for k, v in label_map.items()}
    return reciprocal_label_map[int(label)]


def make_box_from_tensors(
    box_xyxy: torch.Tensor, label: torch.Tensor, score: torch.Tensor
) -> Box:
    x, y, width, height = xyxy_to_xywh(*box_xyxy)
    box = Box(
        label=make_str_label(label),
        x=x,
        y=y,
        width=width,
        height=height,
    )
    box.score = float(score)
    return box


def make_box_list_from_raw_predictions(
    predictions: torch.Tensor,
    nms_iou_threshold: float,
    score_threshold: float,
) -> list[Box]:
    predictions = predictions[0]

    kept_indices = nms(
        boxes=predictions["boxes"],
        scores=predictions["scores"],
        iou_threshold=nms_iou_threshold,
    )
    for k in predictions:
        predictions[k] = predictions[k][kept_indices]

    box_list = []
    for box_xyxy, label, score in zip(
        predictions["boxes"],
        predictions["labels"],
        predictions["scores"],
    ):
        if score < score_threshold:
            continue
        box = make_box_from_tensors(box_xyxy, label, score)
        box_list.append(box)
    return box_list


def make_predictions(
    model: torch.nn.Module,
    image_array: torch.Tensor,
    image_name: str,
    nms_iou_threshold: float,
    score_threshold: float,
) -> Image:
    _, _, height, width = image_array.shape

    predictions = model(image_array.cuda())

    box_list = make_box_list_from_raw_predictions(
        predictions, nms_iou_threshold, score_threshold
    )
    return Image(
        name=image_name,
        width=width,
        height=height,
        annotations=box_list,
    )
