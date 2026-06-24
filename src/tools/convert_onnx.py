import os
from typing import Dict, List

import torch
import yaml

from src.core.model import MultiLabelClassifyModel
from src.entrypoints.bootstrap import create_backbone

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH: str = "outputs/train/v2.resnet50.rap2+cia+cia_gen.change_class_order/weights/best-epoch=34-val_f1_macro=0.533.ckpt"
OUTPUT_PATH: str = "outputs/convert_onnx/v2.resnet50.rap2+cia+cia_gen.change_class_order.onnx"
DUMMY_INPUT: torch.Tensor = torch.randn(1, 3, 256, 192)
INPUT_NAMES: List[str] = ["input"]
OUTPUT_NAMES: List[str] = ["output"]
DYNAMIC_AXES: Dict[str, Dict[int, str]] = {
    "input": {0: "batch"},
    "output": {0: "batch"},
}
# =====================================================

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found at: {CKPT_PATH}")

print(f"Loading model checkpoint from {CKPT_PATH}...")

# 1. Load Config (tìm config.yaml ở thư mục cha của checkpoint)
run_dir = os.path.dirname(os.path.dirname(CKPT_PATH))
config_file = os.path.join(run_dir, "config.yaml")

if not os.path.exists(config_file):
    raise FileNotFoundError(f"Config file not found at {config_file}")

with open(config_file, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

backbone = create_backbone(cfg)

# Load model from checkpoint
model = MultiLabelClassifyModel.load_from_checkpoint(CKPT_PATH, model=backbone)
model.eval()

# Move model to CPU for exporting
model.to("cpu")

# Ensure parent directory of output_path exists
parent_dir: str = os.path.dirname(os.path.abspath(OUTPUT_PATH))
os.makedirs(parent_dir, exist_ok=True)

print(f"Exporting model to ONNX format at {OUTPUT_PATH}...")
torch.onnx.export(
    model,
    DUMMY_INPUT,
    OUTPUT_PATH,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=INPUT_NAMES,
    output_names=OUTPUT_NAMES,
    dynamic_axes=DYNAMIC_AXES,
)
print("ONNX conversion completed successfully!")
