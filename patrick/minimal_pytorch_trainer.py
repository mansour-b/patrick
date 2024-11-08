import torch
from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
from effdet.config.model_config import efficientdet_model_param_dict
from effdet.efficientdet import HeadNet
from torch.utils.data import DataLoader

from patrick.box_detection import load_images
from patrick.efficientdet import EfficientDetDataset


def collate_fn(batch):
    images, targets, image_ids = tuple(zip(*batch))
    images = torch.stack(images)
    images = images.float()

    boxes = [target["bboxes"].float() for target in targets]
    labels = [target["labels"].float() for target in targets]
    img_size = torch.tensor([target["img_size"] for target in targets]).float()
    img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

    annotations = {
        "bbox": boxes,
        "cls": labels,
        "img_size": img_size,
        "img_scale": img_scale,
    }

    return images, annotations, targets, image_ids


def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    efficientdet_model_param_dict["tf_efficientnetv2_l"] = dict(
        name="tf_efficientnetv2_l",
        backbone_name="tf_efficientnetv2_l",
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url="",
    )

    config = get_efficientdet_config(architecture)
    config.update({"num_classes": num_classes})
    config.update({"image_size": (image_size, image_size)})

    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)


if __name__ == "__main__":
    image_list = load_images(annotation_file_path, image_size, polyline_to_box_padding)

    train_dataset = EfficientDetDataset(image_list, image_dir_name, label_map)
    val_dataset = EfficientDetDataset(image_list, image_dir_name, label_map)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    model = create_model(num_classes, image_size, architecture)
