"""
Script đánh giá mô hình phân loại multilabel dưới dạng phân loại single-label.
Script này thực hiện:
- Load dữ liệu validation và model từ file cấu hình (config.yaml) của thư mục run tương ứng với checkpoint.
- Đánh giá mô hình, kiểm tra ràng buộc (assert) đảm bảo mỗi ảnh chỉ có tối đa 1 nhãn Ground Truth (tổng nhãn target active <= 1).
- Dự đoán nhãn cho ảnh bằng cách lấy class có xác suất cao nhất và phải >= 0.5. Nếu tất cả các class đều dưới 0.5 thì coi như không có nhãn ("No Label").
- Tính toán và vẽ Confusion Matrix được chuẩn hóa theo hàng (row-wise normalized).
- Hiển thị trên mỗi ô (cell) của Confusion Matrix các giá trị: tỉ lệ chuẩn hóa, số lượng mẫu thực tế trong ngoặc đơn, và xác suất trung bình của nhãn Ground Truth tại ô đó.
"""

import os
from typing import Any, Iterable, List, TypeVar
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.core.data import MultiLabelDataModule
from src.core.model import MultiLabelClassifyModel
from src.entrypoints.bootstrap import create_backbone, create_transform
from tqdm import tqdm

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = "outputs/train/v22.efficientnetv2.cluster-CNN-8--group-cls-123--cut-l4.remove-tkcsp-ctvgxh/weights/best-epoch=03-val_f1_macro=0.482.ckpt"
# =====================================================


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    save_path: str,
    prob_sums: List[List[List[float]]]
) -> None:
    """Plots and saves a beautiful confusion matrix heatmap."""
    # Row-wise normalization (divide by sum of each row)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title("Normalized Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    num_classes = len(classes) - 1
    thresh = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            # Average probability calculation for this cell
            if i < num_classes and len(prob_sums[i][j]) > 0:
                avg_prob = sum(prob_sums[i][j]) / len(prob_sums[i][j])
                text = f"{val:.2f}\n({cm[i, j]})\n{avg_prob:.2f}"
            else:
                text = f"{val:.2f}\n({cm[i, j]})"

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=10,
            )

    plt.tight_layout()
    plt.ylabel("Ground Truth", fontsize=12, labelpad=10)
    plt.xlabel("Prediction", fontsize=12, labelpad=10)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def evaluate_confusion_matrix(ckpt_path: str) -> None:
    """Evaluates the model and computes the confusion matrix on the validation set."""
    # 1. Load Config (tìm config.yaml ở thư mục cha của checkpoint)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes: List[str] = cfg["classes"]
    num_classes: int = len(classes)
    ckpt_dir = os.path.dirname(ckpt_path)

    # 2. Data Module
    val_transform = create_transform(cfg["val_augment"])

    dm = MultiLabelDataModule(
        train_files=[],
        val_files=cfg["val_data"],
        classes=classes,
        val_transform=val_transform,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    dm.setup(stage="test")
    val_loader = dm.test_dataloader()

    # 3. Load Model
    backbone = create_backbone(cfg)
    model = MultiLabelClassifyModel.load_from_checkpoint(ckpt_path, model=backbone)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds: List[int] = []
    all_gts: List[int] = []
    all_probs: List[List[float]] = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            imgs, targets, paths = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            for i in range(len(probs)):
                prob_i = probs[i]
                gt_i = targets[i]

                # Assert each image has 0 or 1 labels
                gt_sum = gt_i.sum().item()
                assert gt_sum <= 1.0, (
                    f"Ground truth has more than 1 label (sum = {gt_sum}) for image {paths[i]}. "
                    f"Labels: {gt_i.cpu().numpy()}"
                )

                if gt_sum == 0.0:
                    gt_label = num_classes  # "No Label" index
                else:
                    gt_label = int(torch.argmax(gt_i).item())

                max_prob, max_idx = torch.max(prob_i, dim=0)
                if max_prob.item() >= 0.5:
                    pred_label = int(max_idx.item())
                else:
                    pred_label = num_classes  # "No Label"

                all_preds.append(pred_label)
                all_gts.append(gt_label)
                all_probs.append(prob_i.cpu().numpy().tolist())

    # 4. Calculate confusion matrix and collect probabilities
    classes_with_none = classes + ["No Label"]
    num_classes_with_none = len(classes_with_none)

    cm = np.zeros((num_classes_with_none, num_classes_with_none), dtype=np.int32)
    prob_sums: List[List[List[float]]] = [
        [[] for _ in range(num_classes_with_none)]
        for _ in range(num_classes_with_none)
    ]
    for gt, pred, prob in zip(all_gts, all_preds, all_probs):
        cm[gt, pred] += 1
        if gt < num_classes:
            prob_sums[gt][pred].append(prob[gt])

    # 5. Save results
    img_path = os.path.join(run_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes_with_none, img_path, prob_sums)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    evaluate_confusion_matrix(CKPT_PATH)
