from __future__ import annotations

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

from patrick import Box
from patrick.adapters.nn_model import TorchNNModel


class FasterRCNNModel(TorchNNModel):

    def __init__(
        self,
        label_map: dict[str, int],
        nms_iou_threshold: float,
        score_threshold: float,
    ):
        self.label_map = label_map
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold

        self.net = self.get_net(num_classes=len(label_map) + 1)

    @property
    def reversed_label_map(self):
        return {v: k for k, v in self.label_map.items()}

    def get_net(self, num_classes: int):
        net = fasterrcnn_resnet50_fpn()
        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return net

    def convert_predictions(self, predictions: list[dict[torch.Tensor]]) -> list[Box]:
        predictions = predictions[0]

        kept_indices = nms(
            boxes=predictions["boxes"],
            scores=predictions["scores"],
            iou_threshold=self.nms_iou_threshold,
        )
        for k in predictions:
            predictions[k] = predictions[k][kept_indices]

        box_list = []
        for box_xyxy, label, score in zip(
            predictions["boxes"],
            predictions["labels"],
            predictions["scores"],
        ):
            if score < self.score_threshold:
                continue
            box = self.make_box_from_tensors(box_xyxy, label, score)
            box_list.append(box)
        return box_list

    def make_box_from_tensors(
        self,
        box_xyxy: torch.Tensor,
        label: torch.Tensor,
        score: torch.Tensor,
    ) -> Box:
        x, y, width, height = self.xyxy_to_xywh(*box_xyxy)
        return Box(
            label=self.reversed_label_map[int(label)],
            x=x,
            y=y,
            width=width,
            height=height,
            score=score,
        )

    @staticmethod
    def xyxy_to_xywh(
        xmin: float, ymin: float, xmax: float, ymax: float
    ) -> tuple[float, float, float, float]:
        x = xmin
        y = ymin
        width = xmax - xmin
        height = ymax - ymin
        return x, y, width, height
