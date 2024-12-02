from patrick.entities.annotation import Box


def test_create_box():
    box = Box(label="blob", x=0, y=0, width=1, height=1, score=1)
    assert box.label == "glob"
