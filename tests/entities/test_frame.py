from patrick.entities.annotation import Box, Keypoint
from patrick.entities.frame import Frame


class TestFrame:
    def test_init(self):
        frame = Frame(name="frame_0", width=512, height=512, annotations=[])
        assert frame.name == "frame_0"
        assert frame.width == 512
        assert frame.height == 512
        assert frame.annotations == []

    def test_from_dict(self):
        frame_as_dict = {
            "name": "frame_0",
            "width": 512,
            "height": 512,
            "annotations": [
                {
                    "height": 1.0,
                    "label": "blob",
                    "score": 1.0,
                    "type": "box",
                    "width": 1.0,
                    "x": 0.0,
                    "y": 0.0,
                },
            ],
        }
        frame = Frame.from_dict(frame_as_dict)
        assert frame.name == "frame_0"
        assert frame.width == 512
        assert frame.height == 512
        assert frame.annotations == [
            Box(label="blob", score=1.0, x=0.0, y=0.0, width=1.0, height=1.0)
        ]

    def test_to_dict(self):
        frame = Frame(
            name="frame_0",
            width=512,
            height=512,
            annotations=[Box(label="blob", x=0, y=0, width=1, height=1, score=1)],
        )
        assert frame.to_dict() == {
            "name": "frame_0",
            "width": 512,
            "height": 512,
            "annotations": [
                {
                    "height": 1.0,
                    "label": "blob",
                    "score": 1.0,
                    "type": "box",
                    "width": 1.0,
                    "x": 0.0,
                    "y": 0.0,
                },
            ],
        }
