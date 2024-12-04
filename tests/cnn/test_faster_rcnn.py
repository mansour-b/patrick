import torch

from patrick import Box
from patrick.cnn.faster_rcnn import FasterRCNNModel


class MockNet:
    def __call__(self, frame):
        pass


class TestFasterRCNN:
    @staticmethod
    def get_model():
        return FasterRCNNModel(
            label_map={"blob": 1}, nms_iou_threshold=0.2, score_threshold=0.7
        )

    def test_init(self):
        model = self.get_model()
        assert model.label_map == {"blob": 1}
        assert model.nms_iou_threshold == 0.2
        assert model.score_threshold == 0.7

    def test_reversed_label_map(self):
        model = self.get_model()
        assert model.reversed_label_map == {1: "blob"}

    def test_xyxy_to_xywh(self):
        model = self.get_model()
        assert model.xyxy_to_xywh(1, 2, 3, 4) == (1, 2, 2, 2)

    def test_make_box_from_tensors(self):
        model = self.get_model()
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

    def test_post_process(self):
        model = self.get_model()
        predictions = [
            {
                "boxes": torch.tensor([[1, 2, 3, 4]], dtype=float),
                "labels": torch.tensor([1], dtype=int),
                "scores": torch.tensor([1.0], dtype=float),
            }
        ]
        assert model.post_process(predictions) == [
            Box(label="blob", x=1, y=2, width=2, height=2, score=1)
        ]
