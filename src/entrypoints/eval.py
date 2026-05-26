import os
import yaml
import pytorch_lightning as pl
from src.core.model import MultiLabelClassifyModel
from src.core.data import MultiLabelDataModule

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = "runs/classification/multilabel/v1/weights/best-epoch=00-val_f1_macro=0.000.ckpt"
# =====================================================


def evaluate_model(ckpt_path: str) -> None:
    # 1. Load Config (tìm config.yaml ở thư mục cha của checkpoint)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    accelerator = cfg["accelerator"] if "accelerator" in cfg else "gpu"
    devices = cfg["devices"] if "devices" in cfg else "auto"
    num_classes = len(cfg["classes"])

    # 2. Data Module
    dm = MultiLabelDataModule(
        train_files=[],
        val_files=cfg["val_data"],
        num_classes=num_classes,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"] if "num_workers" in cfg else 4,
    )

    # 3. Load Model
    model = MultiLabelClassifyModel.load_from_checkpoint(ckpt_path)
    model.class_names = cfg["classes"]

    # 4. Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=run_dir,
    )

    # 5. Evaluate
    print(f"Evaluating checkpoint: {ckpt_path}")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    evaluate_model(CKPT_PATH)
