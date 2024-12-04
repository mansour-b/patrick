from __future__ import annotations

import torch

from patrick import Frame, NNModel


class TorchNNModel(NNModel):
    net: torch.nn.Module
    nms_iou_threshold: float
    score_threshold: float
    label_map: dict[str, int]

    def predict(self, frame: Frame) -> Frame:
        predictions = self.net(frame.image_array)

        annotations = self.convert_predictions(predictions)

        return Frame(
            name=frame.name,
            width=frame.width,
            height=frame.height,
            annotations=annotations,
        )
