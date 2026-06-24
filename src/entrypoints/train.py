import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from src.core.data import MultiLabelDataModule
from src.core.model import MultiLabelClassifyModel
from src.core.utils import get_run_dir, visualize_batch
from src.entrypoints.bootstrap import create_backbone, create_transform

# ================= CẤU HÌNH TRỰC TIẾP =================
CONFIG_PATH = "configs/test.yaml"  # Path tới file cấu hình
# =====================================================


def train_model(config_path: str) -> None:
    # 1. Load Initial Config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resume_path = cfg["resume_path"]
    accelerator = cfg["accelerator"]
    devices = cfg["devices"]

    backbone = create_backbone(cfg)
    train_transform = create_transform(cfg["train_augment"])
    val_transform = create_transform(cfg["val_augment"])

    # 2. Setup Run Directory & Load Actual Config (if resume)
    if resume_path:
        print(f"Resuming from {resume_path}")
        run_dir = resume_path
        config_file = os.path.join(resume_path, "config.yaml")
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg["devices"] = devices
            cfg["accelerator"] = accelerator
    else:
        run_dir = get_run_dir(cfg["output_dir"])
        # Lưu lại config để phục vụ resume sau này
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True)

    print(f"Output directory: {run_dir}")

    # 3. Data Module
    dm = MultiLabelDataModule(
        train_files=cfg["train_data"],
        val_files=cfg["val_data"],
        classes=cfg["classes"],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    dm.setup(stage="fit")

    # Visualize 3 batches
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        visualize_batch(
            batch,
            cfg["classes"],
            os.path.join(run_dir, f"train_batch_{i}.jpg"),
            f"Train Batch {i}",
        )

    for i, batch in enumerate(val_loader):
        if i >= 3:
            break
        visualize_batch(
            batch,
            cfg["classes"],
            os.path.join(run_dir, f"val_batch_{i}.jpg"),
            f"Val Batch {i}",
        )

    # 4. Model
    model = MultiLabelClassifyModel(
        model=backbone,
        num_classes=len(cfg["classes"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        class_names=cfg["classes"],
    )

    # 5. Callbacks & Loggers
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "weights"),
        filename="best-{epoch:02d}-{val_f1_macro:.3f}",
        monitor="val_f1_macro",
        mode="max",
        save_last=True,
    )

    loggers = [
        CSVLogger(run_dir, name="logs"),
    ]

    patience = cfg["patience"]
    precision = cfg["precision"]

    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_f1_macro", mode="max", patience=patience),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=loggers,
        default_root_dir=run_dir,
        precision=precision,
    )

    # 7. Fit
    ckpt_path = os.path.join(run_dir, "weights", "last.ckpt") if resume_path else None
    if ckpt_path and not os.path.exists(ckpt_path):
        print(f"Warning: last.ckpt not found at {ckpt_path}. Training from scratch.")
        ckpt_path = None

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    train_model(CONFIG_PATH)
