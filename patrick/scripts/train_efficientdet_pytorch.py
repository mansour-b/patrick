import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer

from patrick import PATRICK_DIR_PATH
from patrick.data.image import Image
from patrick.data.operations import deserialise_image_list, polyline_to_box
from patrick.data_module import EfficientDetDataModule
from patrick.training_loop import EfficientDetModel


def load_images(
    annotation_file_path: Path, image_size: int, polyline_to_box_padding: float
) -> list[Image]:
    with open(annotation_file_path) as f:
        image_list = deserialise_image_list(json.load(f))

    for image in image_list:
        image.resize(image_size, image_size)

    image_list = [
        Image(
            name=image._name,
            width=image._width,
            height=image._height,
            annotations=[
                polyline_to_box(
                    polyline,
                    w_padding=polyline_to_box_padding,
                    h_padding=polyline_to_box_padding,
                )
                for polyline in image._annotations
            ],
        )
        for image in image_list
    ]
    return image_list


def make_parser():
    parser = ArgumentParser()
    # experiment
    # image_size
    # polyline_to_box_padding
    # num_workers 4
    # batch_size 2
    # prediction_confidence_threshold 0.2
    # learning_rate 0.0002
    # wbf_iou_threshold 0.44
    # model_architecture tf_efficientnetv2_l
    # num_nodes
    # max_epochs
    # num_sanity_val_steps
    return parser


if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()
    experiment = args.experiment

    annotation_file_path = PATRICK_DIR_PATH / f"annotations/{experiment}.json"
    image_list = load_images(
        annotation_file_path,
        image_size=args.image_size,
        polyline_to_box_padding=args.polyline_to_box_padding,
    )

    label_map = {"blob_front": 1}

    timestamp = time.strftime("%y%m%d_%H%M%S")
    model_name = f"efficientdet_{timestamp}"
    model_path = PATRICK_DIR_PATH / f"models/{model_name}"

    data_module = EfficientDetDataModule(
        train_image_list=image_list,
        val_image_list=image_list,
        image_dir_name=experiment,
        label_map=label_map,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    model = EfficientDetModel(
        num_classes=len(label_map),
        img_size=args.image_size,
        prediction_confidence_threshold=args.prediction_confidence_threshold,
        learning_rate=args.learning_rate,
        wbf_iou_threshold=args.wbf_iou_threshold,
        model_architecture=args.model_architecture,
    )
    trainer = Trainer(
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
    )
    trainer.fit(model, data_module)

    torch.save(model.state_dict(), model_path)
