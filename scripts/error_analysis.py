import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.data import MultiLabelDataset
from src.core.model import MultiLabelClassifyModel
from src.entrypoints.bootstrap import create_backbone, create_transform


def create_image_symlinks(
    cases: List[Tuple[str, float]],
    target_dir: str,
) -> None:
    """Creates symlinks for the given image cases inside target_dir.

    The symlinks are named with the probability score prefix for sorting.
    """
    os.makedirs(target_dir, exist_ok=True)
    # Clear existing files in target_dir to prevent stale errors
    for f in os.listdir(target_dir):
        fp: str = os.path.join(target_dir, f)
        if os.path.islink(fp) or os.path.isfile(fp):
            os.remove(fp)

    for path, p_val in cases:
        img_name: str = os.path.basename(path)
        link_name: str = f"{p_val:.4f}_{img_name}"
        link_path: str = os.path.join(target_dir, link_name)
        try:
            # os.symlink(os.path.realpath(path), link_path)
            shutil.copy2(path, link_path)

        except Exception:
            # Ignore or copy if OS/filesystem doesn't support symlink
            pass


def analyze_errors(
    ckpt_path: str,
    output_dir: str,
    threshold: float = 0.5,
    data_files: Optional[List[str]] = None,
) -> None:
    """Performs error analysis on the validation/test dataset for a given checkpoint.

    Categorizes each prediction per class into TP, FP, and FN, and writes the list
    of image paths and probabilities to output text files under output_dir.

    Args:
        ckpt_path: Path to the model checkpoint.
        output_dir: Directory where the error analysis files will be saved.
        threshold: Decision threshold for predictions.
        data_files: Optional custom list of txt files containing annotation paths.
    """
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Config
    run_dir: str = os.path.dirname(os.path.dirname(ckpt_path))
    config_file: str = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    classes: List[str] = cfg["classes"]

    # 2. Data Module / Loader
    val_transform = create_transform(cfg["val_augment"])

    # If data_files is not provided, use the list from config
    if data_files is None:
        data_files = cfg["val_data"]

    # We use validation data for error analysis
    dataset = MultiLabelDataset(
        data_files=data_files,
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

    # 3. Load Model
    backbone = create_backbone(cfg)
    model = MultiLabelClassifyModel.load_from_checkpoint(ckpt_path, model=backbone).to(
        device
    )
    model.eval()

    # Initialize error lists for each class
    # Keys will be class names, values will be dicts containing lists of tuples (path, prob)
    tp_lists: Dict[str, List[Tuple[str, float]]] = {c: [] for c in classes}
    fp_lists: Dict[str, List[Tuple[str, float]]] = {c: [] for c in classes}
    fn_lists: Dict[str, List[Tuple[str, float]]] = {c: [] for c in classes}

    print("Running inference on validation dataset...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            imgs, targets, paths = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = targets.cpu().numpy()

            for idx in range(len(paths)):
                path: str = paths[idx]
                prob: np.ndarray = probs[idx]
                target: np.ndarray = targets[idx]

                for c_idx, c_name in enumerate(classes):
                    p_val: float = float(prob[c_idx])
                    t_val: float = float(target[c_idx])
                    pred_val: int = 1 if p_val >= threshold else 0

                    if t_val == 1.0:
                        if pred_val == 1:
                            tp_lists[c_name].append((path, p_val))
                        else:
                            fn_lists[c_name].append((path, p_val))
                    else:  # t_val == 0.0
                        if pred_val == 1:
                            fp_lists[c_name].append((path, p_val))

    # 4. Save results to text files
    print(f"\nSaving error analysis results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for console output table
    summary_data: List[Tuple[str, int, int, int]] = []

    for c_name in classes:
        class_dir: str = os.path.join(output_dir, c_name)
        os.makedirs(class_dir, exist_ok=True)

        tps: List[Tuple[str, float]] = tp_lists[c_name]
        fps: List[Tuple[str, float]] = fp_lists[c_name]
        fns: List[Tuple[str, float]] = fn_lists[c_name]

        summary_data.append((c_name, len(tps), len(fps), len(fns)))

        # Sort lists by confidence
        sorted_tps: List[Tuple[str, float]] = sorted(
            tps, key=lambda x: x[1], reverse=True
        )
        sorted_fps: List[Tuple[str, float]] = sorted(
            fps, key=lambda x: x[1], reverse=True
        )
        sorted_fns: List[Tuple[str, float]] = sorted(fns, key=lambda x: x[1])

        # Save TP list
        tp_file: str = os.path.join(class_dir, "TP.txt")
        with open(tp_file, "w", encoding="utf-8") as f:
            for path, p_val in sorted_tps:
                f.write(f"{path},{p_val:.4f}\n")

        # Save FP list (sorted by confidence descending - most confident errors first)
        fp_file: str = os.path.join(class_dir, "FP.txt")
        with open(fp_file, "w", encoding="utf-8") as f:
            for path, p_val in sorted_fps:
                f.write(f"{path},{p_val:.4f}\n")

        # Save FN list (sorted by confidence ascending - most confident omissions first)
        fn_file: str = os.path.join(class_dir, "FN.txt")
        with open(fn_file, "w", encoding="utf-8") as f:
            for path, p_val in sorted_fns:
                f.write(f"{path},{p_val:.4f}\n")

        # Create image directories and symlinks
        create_image_symlinks(sorted_tps, os.path.join(class_dir, "TP_images"))
        create_image_symlinks(sorted_fps, os.path.join(class_dir, "FP_images"))
        create_image_symlinks(sorted_fns, os.path.join(class_dir, "FN_images"))

    # 5. Print Summary Table
    print("\n" + "=" * 95)
    print(
        f"{'Class Name':<25} | {'TP':<6} | {'FP':<6} | {'FN':<6} | "
        f"{'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}"
    )
    print("-" * 95)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for c_name, tp_count, fp_count, fn_count in summary_data:
        precision: float = (
            tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        )
        recall: float = (
            tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        )
        f1: float = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(
            f"{c_name:<25} | {tp_count:<6} | {fp_count:<6} | {fn_count:<6} | "
            f"{precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}"
        )

    print("-" * 95)
    macro_precision: float = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall: float = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1: float = sum(f1s) / len(f1s) if f1s else 0.0
    print(
        f"{'Macro Average':<25} | {'-':<6} | {'-':<6} | {'-':<6} | "
        f"{macro_precision:<10.4f} | {macro_recall:<10.4f} | {macro_f1:<10.4f}"
    )
    print("=" * 95)
    print("Error analysis completed successfully.")


if __name__ == "__main__":
    # Standard paths
    ckpt_path_default: str = (
        "outputs/train/v6.resnet50.pa100k/weights/best-epoch=03-val_f1_macro=0.664.ckpt"
    )
    output_dir_default: str = "outputs/error_analysis"

    # Custom list of annotation files for error analysis
    custom_data_files: Optional[List[str]] = [
        # "/home/laptq/data/processed/fs26/train_data/classify_rgb_multilabel/person_attributes/pa100k_val.txt",
        # "/home/laptq/data/processed/fs26/train_data/classify_rgb_multilabel/person_attributes/pa100k_test.txt",
        "/home/laptq/classification_multilabel/outputs/trivials/filtered_Satudora_test.txt"
    ]

    # Allow overriding via environment variables or use defaults
    ckpt: str = (
        os.environ["CKPT_PATH"] if "CKPT_PATH" in os.environ else ckpt_path_default
    )
    out: str = (
        os.environ["OUTPUT_DIR"] if "OUTPUT_DIR" in os.environ else output_dir_default
    )

    analyze_errors(ckpt, out, data_files=custom_data_files)
