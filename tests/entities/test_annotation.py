from patrick.entities.annotation import Box


def test_create_box():
    box = Box(label="blob")
    assert box.label == "glob"
