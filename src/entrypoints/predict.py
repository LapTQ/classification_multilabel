import os
import yaml
from typing import List, Tuple
import torch
from PIL import Image
import torchvision.transforms as T
from src.core.model import MultiLabelClassifyModel
from src.core.augmentations import SquarePad

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH = "runs/classification/multilabel/v1/weights/best-epoch=00-val_f1_macro=0.000.ckpt"
INPUT_PATH = "data/test_images"  # Path tới file ảnh, file .txt hoặc thư mục
OUTPUT_PATH = "predictions.txt"
THRESHOLD = 0.5
# =====================================================


def predict_model(
    ckpt_path: str, input_path: str, output_path: str, threshold: float = 0.5
) -> None:
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model & Config
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    config_file = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg["classes"]
    model = MultiLabelClassifyModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    # 3. Transform
    transform = T.Compose(
        [
            SquarePad(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

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
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                logits = model(img_tensor)
                probs = torch.sigmoid(logits)[0]  # Sigmoid for multilabel

                active_preds = []
                for idx, prob_val in enumerate(probs):
                    prob = float(prob_val.item())
                    if prob >= threshold:
                        active_preds.append(f"{classes[idx]}:{prob:.4f}")

                if not active_preds:
                    # Fallback to the class with the highest probability
                    top_idx = int(torch.argmax(probs).item())
                    top_prob = float(probs[top_idx].item())
                    active_preds.append(
                        f"{classes[top_idx]}:{top_prob:.4f} (fallback)"
                    )

                pred_str = ",".join(active_preds)
                results.append((path, pred_str))
            except Exception as e:
                print(f"Error predicting {path}: {e}")

    # 6. Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for p, pred_str in results:
            f.write(f"{p}\t{pred_str}\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    predict_model(CKPT_PATH, INPUT_PATH, OUTPUT_PATH, THRESHOLD)
