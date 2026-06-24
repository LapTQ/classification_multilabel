import os
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MultiLabelDataset(Dataset):
    def __init__(
        self,
        data_files: Union[str, List[str]],
        classes: List[str],
        transform: Optional[T.Compose] = None,
    ) -> None:
        """
        Args:
            data_files: Path or list of paths to data txt files.
            classes: List of target class names.
            transform: torchvision transforms.
        """
        self.image_paths: List[str] = []
        self.targets: List[np.ndarray] = []
        self.transform: Optional[T.Compose] = transform
        self.classes: List[str] = classes
        self.num_classes: int = len(classes)

        class_to_idx = {name.strip().lower(): idx for idx, name in enumerate(classes)}

        if isinstance(data_files, str):
            data_files = [data_files]

        for txt_file in data_files:
            if not os.path.exists(txt_file):
                print(f"Warning: File {txt_file} does not exist.")
                continue
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    img_path = parts[0]
                    if not os.path.exists(img_path):
                        # Warning to avoid spamming stdout for thousands of missing images
                        # but keep it informative
                        continue

                    # Parse labels as class names
                    labels = []
                    for p in parts[1:]:
                        cls_name = p.strip().lower()
                        if cls_name:
                            if cls_name in class_to_idx:
                                labels.append(class_to_idx[cls_name])

                    # Convert to multi-hot target
                    target = np.zeros(self.num_classes, dtype=np.float32)
                    for label in labels:
                        if 0 <= label < self.num_classes:
                            target[label] = 1.0

                    self.image_paths.append(img_path)
                    self.targets.append(target)

        # Shuffle dataset
        idxs = list(range(len(self.image_paths)))
        random.seed(42)
        random.shuffle(idxs)
        self.image_paths = [self.image_paths[i] for i in idxs]
        self.targets = [self.targets[i] for i in idxs]

        print(
            f"Dataset initialized with {len(self.image_paths)} images across {self.num_classes} classes."
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[index]
        target = self.targets[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(target, dtype=torch.float32), img_path


class MultiLabelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_files: Union[str, List[str]],
        val_files: Union[str, List[str]],
        classes: List[str],
        train_transform: Optional[T.Compose] = None,
        val_transform: Optional[T.Compose] = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_files: Union[str, List[str]] = train_files
        self.val_files: Union[str, List[str]] = val_files
        self.classes: List[str] = classes
        self.num_classes: int = len(classes)
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.train_ds: Optional[MultiLabelDataset] = None
        self.val_ds: Optional[MultiLabelDataset] = None
        self.test_ds: Optional[MultiLabelDataset] = None
        self.predict_ds: Optional[MultiLabelDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_ds = MultiLabelDataset(
                self.train_files, self.classes, self.train_transform
            )
            self.val_ds = MultiLabelDataset(
                self.val_files, self.classes, self.val_transform
            )
        if stage == "test":
            self.test_ds = MultiLabelDataset(
                self.val_files, self.classes, self.val_transform
            )
        if stage == "predict":
            self.predict_ds = MultiLabelDataset(
                self.val_files, self.classes, self.val_transform
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise ValueError("train_ds is not initialized. Call setup() first.")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise ValueError("val_ds is not initialized. Call setup() first.")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise ValueError("test_ds is not initialized. Call setup() first.")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.predict_ds is None:
            raise ValueError("predict_ds is not initialized. Call setup() first.")
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
