import os
from typing import Dict, List
import torch
from src.core.model import MultiLabelClassifyModel

# ================= CẤU HÌNH TRỰC TIẾP =================
CKPT_PATH: str = "outputs/train/v2.resnet50.rap2+cia+cia_gen.change_class_order/weights/best-epoch=34-val_f1_macro=0.533.ckpt"
OUTPUT_PATH: str = "outputs/convert_onnx/v2.resnet50.rap2+cia+cia_gen.change_class_order.onnx"  # Để trống để tự động lưu cùng thư mục checkpoint
DUMMY_INPUT: torch.Tensor = torch.randn(1, 3, 256, 192)
INPUT_NAMES: List[str] = ["input"]
OUTPUT_NAMES: List[str] = ["output"]
DYNAMIC_AXES: Dict[str, Dict[int, str]] = {
    "input": {0: "batch"},  # cho phép thay đổi batch size động
    "output": {0: "batch"},
}
# =====================================================

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found at: {CKPT_PATH}")

print(f"Loading model checkpoint from {CKPT_PATH}...")
# Load model from checkpoint
model = MultiLabelClassifyModel.load_from_checkpoint(CKPT_PATH)
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
