import glob
import os
from typing import List
import matplotlib.pyplot as plt
import pandas as pd


def visualize_training(run_dir: str) -> None:
    # Tìm file metrics.csv (thường ở logs/version_0/metrics.csv)
    csv_pattern: str = os.path.join(run_dir, "logs", "version_*", "metrics.csv")
    csv_files: List[str] = glob.glob(csv_pattern)

    if not csv_files:
        print(f"Error: Không tìm thấy file metrics.csv trong {run_dir}")
        return

    # Lấy file metrics mới nhất (thường chỉ có 1 version)
    csv_path: str = max(csv_files, key=os.path.getmtime)
    print(f"Đang đọc dữ liệu từ: {csv_path}")

    df: pd.DataFrame = pd.read_csv(csv_path)

    # Sửa lỗi: Điền các giá trị epoch bị trống (do Lightning log LR ở hàng riêng ngay trước epoch mới)
    # Dùng bfill() để hàng chứa LR mới được gán đúng vào epoch tiếp theo
    df["epoch"] = df["epoch"].bfill().ffill().fillna(0)

    # Gom dữ liệu theo epoch
    metrics_by_epoch: pd.DataFrame = df.groupby("epoch").max().reset_index()

    epochs = metrics_by_epoch["epoch"]

    plt.figure(figsize=(18, 5))

    # 1. Plot Loss
    plt.subplot(1, 3, 1)
    if "train_loss_epoch" in metrics_by_epoch.columns:
        plt.plot(
            epochs,
            metrics_by_epoch["train_loss_epoch"],
            "-o",
            label="Train Loss",
        )
    elif "train_loss" in metrics_by_epoch.columns:
        plt.plot(
            epochs, metrics_by_epoch["train_loss"], "-o", label="Train Loss"
        )

    if "val_loss" in metrics_by_epoch.columns:
        plt.plot(epochs, metrics_by_epoch["val_loss"], "-o", label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Plot F1-Score / Accuracy
    plt.subplot(1, 3, 2)
    has_f1: bool = False
    if "train_f1_macro" in metrics_by_epoch.columns:
        plt.plot(
            epochs,
            metrics_by_epoch["train_f1_macro"],
            "-o",
            label="Train F1 Macro",
        )
        has_f1 = True
    if "val_f1_macro" in metrics_by_epoch.columns:
        plt.plot(
            epochs,
            metrics_by_epoch["val_f1_macro"],
            "-o",
            label="Val F1 Macro",
        )
        has_f1 = True

    if not has_f1:
        if "train_acc" in metrics_by_epoch.columns:
            plt.plot(
                epochs, metrics_by_epoch["train_acc"], "-o", label="Train Acc"
            )
        if "val_acc" in metrics_by_epoch.columns:
            plt.plot(
                epochs, metrics_by_epoch["val_acc"], "-o", label="Val Acc"
            )
        plt.title("Accuracy Curve")
        plt.ylabel("Accuracy")
    else:
        plt.title("F1-Score Curve")
        plt.ylabel("F1-Score")

    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    # 3. Plot Learning Rate
    plt.subplot(1, 3, 3)
    lr_cols: List[str] = [
        c for c in metrics_by_epoch.columns if c.startswith("lr")
    ]
    if lr_cols:
        # Lấy cột LR đầu tiên tìm thấy
        plt.plot(
            epochs,
            metrics_by_epoch[lr_cols[0]],
            "-o",
            color="green",
            label="Learning Rate",
        )
    plt.title("Learning Rate Curve")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    output_path: str = os.path.join(run_dir, "training_curves.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu đồ thị tại: {output_path}")


if __name__ == "__main__":
    # Dán đường dẫn thư mục run của bạn vào đây
    RUN_DIR: str = "/home/laptq/laptq-fs26-shoplifting-detection/runs/classification_multilabel/v1.efficientnetv2s.rap2+cia+cia_gen"

    visualize_training(RUN_DIR)
