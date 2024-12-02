from patrick.entities.annotation import Box


class TestBox:

    def test_init(self):
        box = Box(label="blob", x=0, y=0, width=1, height=1, score=1)
        assert box.label == "blob"
        assert box.x == 0
        assert box.y == 0
        assert box.width == 1
        assert box.height == 1
        assert box.score == 1

    def test_xyxy(self):
        box = Box(label="blob", x=0, y=0, width=1, height=1, score=1)
        assert box.xmin == 0
        assert box.ymin == 0
        assert box.xmax == 1
        assert box.ymax == 1

    def test_from_dict(self):
        box_as_dict = {
            "type": "box",
            "label": "blob",
            "x": 0,
            "y": 0,
            "width": 1,
            "height": 1,
            "score": 1,
        }
        box = Box.from_dict(box_as_dict)
        assert box.label == "blob"
        assert box.x == 0
        assert box.y == 0
        assert box.width == 1
        assert box.height == 1
        assert box.score == 1

    def test_to_dict(self):
        box = Box(label="blob", x=0, y=0, width=1, height=1, score=1)
        box_as_dict = box.to_dict()

        assert box_as_dict == {
            "type": "box",
            "label": "blob",
            "x": 0,
            "y": 0,
            "width": 1,
            "height": 1,
            "score": 1,
        }

    def test_rescale(self):
        box = Box(label="blob", x=1, y=1, width=1, height=1, score=1)
        box.rescale(w_ratio=2, h_ratio=3)
        assert box.label == "blob"
        assert box.x == 2
        assert box.y == 3
        assert box.width == 2
        assert box.height == 2
        assert box.score == 1
