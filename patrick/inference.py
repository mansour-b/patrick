import torch

from patrick.data.annotation import Box
from patrick.data.image import Image


def make_batch_image_tensor(
    image_list: list[Image], image_dir_name: str
) -> torch.Tensor:
    output = torch.tensor(
        [image.get_image_array(image_dir_name) for image in image_list]
    )
    output = output.unsqueeze(1).expand(-1, 3, -1, -1)
    return output.float()


def make_box_from_detections(box_coords: list[float], label: str, score: float) -> Box:
    xmin, ymin, xmax, ymax = box_coords
    box = Box(label=label, x=xmin, y=ymin, width=xmax - xmin, height=ymax - ymin)
    box._score = score
    return box


def make_box_list(
    prediction_dict: dict[str, list], label_map: dict[str, int]
) -> list[Box]:
    reciprocal_label_map = {v: k for k, v in label_map.items()}

    box_coords_list = prediction_dict["box_coords"]
    label_list = [
        reciprocal_label_map[int(label)] for label in prediction_dict["label"]
    ]
    score_list = prediction_dict["score"]

    return [
        make_box_from_detections(box_coords, label, score)
        for box_coords, label, score in zip(box_coords_list, label_list, score_list)
    ]


def compute_predictions(
    image_list: list[Image],
    image_dir_name: str,
    model: torch.nn.Module,
    label_map: dict[str:int],
) -> list[Image]:

    batch_image_tensor = make_batch_image_tensor(image_list, image_dir_name)

    pred_bboxes, pred_labels, pred_scores = model.predict(batch_image_tensor)
    big_prediction_list = [
        {"box_coords": box_coords_list, "label": label_list, "score": score_list}
        for box_coords_list, label_list, score_list in zip(
            pred_bboxes, pred_labels, pred_scores
        )
    ]
    big_prediction_dict = {
        image._name: prediction_dict
        for image, prediction_dict in zip(image_list, big_prediction_list)
    }

    output_image_list = []
    for image in image_list:
        prediction_dict = big_prediction_dict[image._name]
        box_list = make_box_list(prediction_dict, label_map)
        pred_image = Image(
            name=image._name,
            width=image._width,
            height=image._height,
            annotations=box_list,
        )
        output_image_list.append(pred_image)

    return output_image_list
