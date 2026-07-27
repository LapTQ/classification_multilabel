import os
from typing import List, Tuple

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from src.core.model import MultiLabelClassifyModel
from src.entrypoints.bootstrap import create_backbone, create_transform

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = (
    "models/checkpoints/fs26/action+attributes+view/classification_multilabel/v1.efficientnetv2s.action_8classes_group123+cia_orig+cia_syn+pa100k+satudora10k/weights/best-epoch=28-val_f1_macro=0.669.ckpt"
)
INPUT_PATH = "data/tmp/remove_dup_path/action.for_CNN.8_classes_grouped_123.cut_left_4_frames.val--min4k--max5k.txt"  # Path tới file ảnh, file .txt hoặc thư mục
OUTPUT_PATH = "/home/laptq/classification_multilabel/data/tmp/predictions.txt"
THRESHOLD = 0.0
DEVICE = "cuda:0"
BATCH_SIZE = 64
# =====================================================


def predict_model(
    ckpt_path: str,
    input_path: str,
    output_path: str,
    threshold: float = 0.5,
    device_str: str = DEVICE,
    batch_size: int = BATCH_SIZE,
) -> None:
    # 1. Setup Device
    device = torch.device(device_str)

    # 2. Load Model & Config
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg["classes"]

    backbone = create_backbone(cfg)
    model = MultiLabelClassifyModel.load_from_checkpoint(ckpt_path, model=backbone).to(
        device
    )
    model.eval()

    # 3. Transform
    transform = create_transform(cfg["val_augment"])

    # 4. Collect Images
    image_paths: List[str] = []
    if os.path.isfile(input_path):
        if input_path.endswith(".txt"):
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        image_paths.append(line.split(",")[0].strip())
        else:
            image_paths = [input_path]
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, f))

    # 5. Predict
    print(f"Predicting {len(image_paths)} images...")
    results: List[Tuple[str, str]] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors: List[torch.Tensor] = []
            valid_paths: List[str] = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = transform(img)
                    batch_tensors.append(img_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")

            if not batch_tensors:
                continue

            # Stack images to form a batch tensor: [B, C, H, W]
            img_batch_tensor = torch.stack(batch_tensors).to(device)
            logits = model(img_batch_tensor)
            probs = torch.sigmoid(logits)  # Sigmoid for multilabel

            for idx_in_batch, path in enumerate(valid_paths):
                item_probs = probs[idx_in_batch]
                active_preds: List[str] = []
                for idx, prob_val in enumerate(item_probs):
                    prob = float(prob_val.item())
                    if prob >= threshold:
                        active_preds.append(f"{classes[idx]}:{prob:.4f}")

                if not active_preds:
                    # Fallback to the class with the highest probability
                    top_idx = int(torch.argmax(item_probs).item())
                    top_prob = float(item_probs[top_idx].item())
                    active_preds.append(
                        f"{classes[top_idx]}:{top_prob:.4f} (fallback)"
                    )

                pred_str = ",".join(active_preds)
                results.append((path, pred_str))

    # 6. Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for p, pred_str in results:
            f.write(f"{p}\t{pred_str}\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    predict_model(CKPT_PATH, INPUT_PATH, OUTPUT_PATH, THRESHOLD, DEVICE, BATCH_SIZE)
