from __future__ import annotations

from typing import TYPE_CHECKING

from patrick.cnn.faster_rcnn import FasterRCNNModel

if TYPE_CHECKING:
    from pathlib import Path

    from patrick.entities.detection import NNModel


def cnn_factory(
    model_type: str,
    net_path: Path,
    label_map: dict[str, int],
    post_processing_parameters: dict,
) -> NNModel:
    model_class = {"faster_rcnn": FasterRCNNModel}[model_type]
    return model_class(net_path, label_map, post_processing_parameters)
