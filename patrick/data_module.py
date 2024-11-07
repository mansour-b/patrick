import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from patrick.data.image import Image
from patrick.efficientdet import EfficientDetDataset


class EfficientDetDataModule(LightningDataModule):

    def __init__(
        self,
        train_image_list: list[Image],
        val_image_list: list[Image],
        image_dir_name: str,
        label_map: dict[str, int],
        num_workers: int = 4,
        batch_size: int = 8,
    ):

        self._train_image_list = train_image_list
        self._val_image_list = val_image_list
        self._image_dir_name = image_dir_name
        self._label_map = label_map
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            image_list=self._train_image_list,
            image_dir_name=self._image_dir_name,
            label_map=self._label_map,
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            image_list=self._val_image_list,
            image_dir_name=self._image_dir_name,
            label_map=self._label_map,
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader

    @staticmethod
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
