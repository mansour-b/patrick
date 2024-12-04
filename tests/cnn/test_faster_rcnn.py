import torch

from patrick import Box
from patrick.cnn.faster_rcnn import FasterRCNNModel


class TestFasterRCNN:

    def test_init(self):
        model = FasterRCNNModel({"blob": 1})
        assert model.label_map == {"blob": 1}

    def test_reversed_label_map(self):
        model = FasterRCNNModel({"blob": 1})
        assert model.reversed_label_map == {1: "blob"}

    def test_xyxy_to_xywh(self):
        model = FasterRCNNModel({"blob": 1})
        assert model.xyxy_to_xywh(1, 2, 3, 4) == (1, 2, 2, 2)

    def test_make_box_from_tensors(self):
        model = FasterRCNNModel({"blob": 1})
        box_xyxy = torch.tensor([1, 2, 3, 4])
        label = torch.tensor(1)
        score = torch.tensor(1.0)
        assert model.make_box_from_tensors(box_xyxy, label, score) == Box(
            label="blob",
            x=1,
            y=2,
            width=2,
            height=2,
            score=1,
        )
