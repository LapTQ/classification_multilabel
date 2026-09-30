import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data import MultiLabelDataset
from src.core.model import MultiLabelClassifyModel
from src.entrypoints.bootstrap import create_backbone, create_transform

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH: str = (
    "models/checkpoints/fs26/action+attributes+view/classification_multilabel/"
    "v5.efficientnetv2s.action_8classes_group123+resagepar_v1_v2+cia_orig+cia_syn+pa100k+satudora10k.with_normal/"
    "weights/best-epoch=22-val_f1_macro=0.716.ckpt"
)

DATA_FILES: list[str] = [
    "data/processed/fs26/action+attributes+view/with_normal/action.for_CNN.8_classes_grouped_123.manually_selected.val_easy.txt",
    "data/processed/fs26/action+attributes+view/with_normal/action.for_CNN.8_classes_grouped_123.manually_selected.val_medium.txt",
]

THRESHOLD: float = 0.5
# =====================================================


def load_model_and_config(
    ckpt_path: str,
    device: torch.device,
) -> tuple[MultiLabelClassifyModel, dict[str, Any]]:
    """Loads configuration and checkpoint weights into the model.

    Args:
        ckpt_path: Path to the model checkpoint.
        device: Torch device to place the model on.

    Returns:
        A tuple of (model, config_dict).
    """
    run_dir: str = os.path.dirname(os.path.dirname(ckpt_path))
    config_file: str = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    backbone = create_backbone(cfg)
    model = MultiLabelClassifyModel.load_from_checkpoint(
        ckpt_path, model=backbone
    ).to(device)
    model.eval()
    return model, cfg


def evaluate_dataset_cases(
    model: MultiLabelClassifyModel,
    cfg: dict[str, Any],
    data_file: str,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluates a single dataset file for the target error condition:

    - GT: not normal AND (cho_tay_vao_tui_quan OR cho_tay_vao_tui_ao_tui_deo_tui_cam_tay)
    - Predict: normal AND (NOT cho_tay_vao_tui_quan AND NOT cho_tay_vao_tui_ao_tui_deo_tui_cam_tay)

    Args:
        model: Evaluated PyTorch Lightning model.
        cfg: Configuration dictionary.
        data_file: Path to dataset annotation txt file.
        device: Torch device for inference.
        threshold: Classification decision threshold.

    Returns:
        Dictionary containing counts and statistics.
    """
    classes: list[str] = cfg["classes"]
    val_transform = create_transform(cfg["val_augment"])

    dataset = MultiLabelDataset(
        data_files=[data_file],
        classes=classes,
        transform=val_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    normal_cls: str = "normal"
    pocket_cls_1: str = "cho_tay_vao_tui_quan"
    pocket_cls_2: str = "cho_tay_vao_tui_ao_tui_deo_tui_cam_tay"

    if normal_cls not in classes:
        raise ValueError(f"Class '{normal_cls}' not found in class list: {classes}")
    if pocket_cls_1 not in classes:
        raise ValueError(f"Class '{pocket_cls_1}' not found in class list: {classes}")
    if pocket_cls_2 not in classes:
        raise ValueError(f"Class '{pocket_cls_2}' not found in class list: {classes}")

    idx_normal: int = classes.index(normal_cls)
    idx_pocket_1: int = classes.index(pocket_cls_1)
    idx_pocket_2: int = classes.index(pocket_cls_2)

    total_samples: int = 0
    gt_target_count: int = 0
    matched_error_count: int = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {os.path.basename(data_file)}"):
            imgs, targets, *rest = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = targets.cpu().numpy()

            for idx in range(len(targets_np)):
                total_samples += 1
                target: np.ndarray = targets_np[idx]
                prob: np.ndarray = probs[idx]

                gt_is_normal: bool = bool(target[idx_normal] == 1.0)
                gt_has_pocket: bool = bool(
                    target[idx_pocket_1] == 1.0 or target[idx_pocket_2] == 1.0
                )

                # GT Condition: không normal, có cho_tay_vao_tui_quan hoặc cho_tay_vao_tui_ao_tui_deo_tui_cam_tay
                gt_condition: bool = (not gt_is_normal) and gt_has_pocket

                if gt_condition:
                    gt_target_count += 1

                    pred_normal: bool = bool(prob[idx_normal] >= threshold)
                    pred_pocket_1: bool = bool(prob[idx_pocket_1] >= threshold)
                    pred_pocket_2: bool = bool(prob[idx_pocket_2] >= threshold)

                    # Predict Condition: có normal, không cho_tay_vao_tui_quan và không cho_tay_vao_tui_ao_tui_deo_tui_cam_tay
                    pred_condition: bool = (
                        pred_normal and (not pred_pocket_1) and (not pred_pocket_2)
                    )

                    if pred_condition:
                        matched_error_count += 1

    return {
        "file": data_file,
        "total_samples": total_samples,
        "gt_target_count": gt_target_count,
        "matched_error_count": matched_error_count,
    }


def main() -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Threshold: {THRESHOLD}")

    model, cfg = load_model_and_config(CKPT_PATH, device)

    total_all_samples: int = 0
    total_gt_target: int = 0
    total_matched: int = 0

    print("\n" + "=" * 105)
    print(
        f"{'Dataset':<55} | {'Total':<8} | {'GT (Target)':<12} | {'Matched Error':<14} | {'Error Rate (%)':<15}"
    )
    print("-" * 105)

    for data_file in DATA_FILES:
        if not os.path.exists(data_file):
            print(f"Warning: File {data_file} not found. Skipping.")
            continue

        res = evaluate_dataset_cases(
            model=model,
            cfg=cfg,
            data_file=data_file,
            device=device,
            threshold=THRESHOLD,
        )

        n_total: int = res["total_samples"]
        n_gt: int = res["gt_target_count"]
        n_matched: int = res["matched_error_count"]
        rate: float = (n_matched / n_gt * 100.0) if n_gt > 0 else 0.0

        total_all_samples += n_total
        total_gt_target += n_gt
        total_matched += n_matched

        fname: str = os.path.basename(data_file)
        print(
            f"{fname:<55} | {n_total:<8} | {n_gt:<12} | {n_matched:<14} | {rate:<15.2f}"
        )

    print("-" * 105)
    overall_rate: float = (
        (total_matched / total_gt_target * 100.0) if total_gt_target > 0 else 0.0
    )
    print(
        f"{'TOTAL / COMBINED':<55} | {total_all_samples:<8} | {total_gt_target:<12} | {total_matched:<14} | {overall_rate:<15.2f}"
    )
    print("=" * 105)


if __name__ == "__main__":
    main()
