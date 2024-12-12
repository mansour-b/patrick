from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import nms

from patrick.core import Box, ComputingDevice, Frame, NeuralNet, NNModel


class FasterRCNNModel(NNModel):
    _device: ComputingDevice

    def __init__(
        self,
        net: NeuralNet,
        label_map: dict[str, int],
        model_parameters: dict,
    ):
        self.net = net
        self.label_map = label_map
        self.pre_proc_params = model_parameters["pre_processing"]
        self.post_proc_params = model_parameters["post_processing"]

    def pre_process(self, frame: Frame) -> torch.Tensor:
        input_array = np.expand_dims(frame.image_array, axis=0)
        input_array = np.repeat(input_array, repeats=3, axis=0)

        input_array = np.expand_dims(input_array, axis=0)

        input_array = torch.as_tensor(input_array)
        return input_array.to(torch.float32).to(self._device)

    def post_process(self, predictions: list[dict[torch.Tensor]]) -> list[Box]:
        predictions = predictions[0]

        kept_indices = nms(
            boxes=predictions["boxes"],
            scores=predictions["scores"],
            iou_threshold=self.post_proc_params["nms_iou_threshold"],
        )
        for k in predictions:
            predictions[k] = predictions[k][kept_indices]

        box_list = []
        for box_xyxy, label, score in zip(
            predictions["boxes"],
            predictions["labels"],
            predictions["scores"],
        ):
            if score < self.post_proc_params["score_threshold"]:
                continue
            box = self.make_box_from_tensors(box_xyxy, label, score)
            box_list.append(box)
        return box_list

    @property
    def reversed_label_map(self):
        return {v: k for k, v in self.label_map.items()}

    def make_box_from_tensors(
        self,
        box_xyxy: torch.Tensor,
        label: torch.Tensor,
        score: torch.Tensor,
    ) -> Box:
        x, y, width, height = self.xyxy_to_xywh(*box_xyxy)
        str_label = self.reversed_label_map[int(label)]
        return Box(label=str_label, x=x, y=y, width=width, height=height, score=score)

    @staticmethod
    def xyxy_to_xywh(
        xmin: float, ymin: float, xmax: float, ymax: float
    ) -> tuple[float, float, float, float]:
        x = xmin
        y = ymin
        width = xmax - xmin
        height = ymax - ymin
        return x, y, width, height
