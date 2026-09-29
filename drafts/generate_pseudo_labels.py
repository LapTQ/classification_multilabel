import os
import sys
from typing import List, Optional, Set, Tuple
from tqdm import tqdm

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Add workspace root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.model import MultiLabelClassifyModel
from src.entrypoints.bootstrap import create_backbone, create_transform

# Config variables
DEVICE: str = "cuda:0"  # Thay đổi thành cuda:5 nếu cần sử dụng card 5
THRESHOLD: float = 0.5


def locate_file(path: str) -> str:
    """Check if the path exists. If not, try alternative path mapping."""
    if os.path.exists(path):
        return path
    # Ánh xạ từ data/tmp/singlelabel_to_multilabel/person_view/ sang data/tmp/person_view_labels/gt/ nếu thiếu
    alt_path: str = path.replace(
        "data/tmp/singlelabel_to_multilabel/person_view/",
        "data/tmp/person_view_labels/gt/",
    )
    if os.path.exists(alt_path):
        return alt_path
    return path


class PseudoLabelDataset(Dataset):
    """Dataset for batch-wise pseudo labeling."""

    def __init__(self, image_paths: List[str], transform: T.Compose) -> None:
        self.image_paths: List[str] = image_paths
        self.transform: T.Compose = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, bool]:
        img_path: str = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        tensor: torch.Tensor = self.transform(img)
        return img_path, tensor, True


def generate_pseudo_labels(
    ckpt_path: str,
    file_list: List[str],
    output_dir: str,
    prefix_to_strip: str,
    allowed_classes: Optional[Set[str]] = None,
    threshold: float = 0.5,
    device_name: str = "cuda:0",
    batch_size: int = 512,
    num_workers: int = 4,
    only_highest_prob: bool = False,
) -> None:
    """Runs inference and generates pseudo label files."""
    device: torch.device = torch.device(
        device_name if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load model configuration
    run_dir: str = os.path.dirname(os.path.dirname(ckpt_path))
    config_file: str = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes: List[str] = cfg["classes"]
    backbone = create_backbone(cfg)
    model = MultiLabelClassifyModel.load_from_checkpoint(
        ckpt_path, model=backbone, map_location=device
    ).to(device)
    model.eval()

    transform = create_transform(cfg["val_augment"])

    for input_file in file_list:
        resolved_file: str = locate_file(input_file)

        # Lấy phần còn lại của đường dẫn để lưu
        rel_path: str = input_file
        if input_file.startswith(prefix_to_strip):
            rel_path = input_file[len(prefix_to_strip) :].lstrip("/")

        output_file: str = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Đọc danh sách ảnh
        image_paths: List[str] = []
        with open(resolved_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    image_paths.append(line.split(",")[0].strip())

        if not image_paths:
            print(f"No images found in {resolved_file}, skipping.")
            continue

        dataset = PseudoLabelDataset(image_paths, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        predictions: dict[str, List[str]] = {}
        print(f"Inferring {len(image_paths)} images from {resolved_file}...")

        with torch.no_grad():
            for paths, batch_tensors, successes in tqdm(dataloader):
                # Chuyển batch_tensors lên GPU
                batch_tensors = batch_tensors.to(device)
                logits = model(batch_tensors)
                probs: torch.Tensor = torch.sigmoid(logits)

                for i, path in enumerate(paths):
                    if not successes[i]:
                        predictions[path] = []
                        continue

                    prob_vals = probs[i].cpu().numpy()
                    active_candidates: List[Tuple[str, float]] = []
                    for cls_idx, prob in enumerate(prob_vals):
                        if prob >= threshold:
                            cls_name = classes[cls_idx]
                            if (
                                allowed_classes is None
                                or cls_name in allowed_classes
                            ):
                                active_candidates.append((cls_name, float(prob)))

                    if only_highest_prob and active_candidates:
                        # Sắp xếp giảm dần theo xác suất
                        active_candidates.sort(key=lambda x: x[1], reverse=True)
                        active_labels: List[str] = [active_candidates[0][0]]
                    else:
                        active_labels = [cand[0] for cand in active_candidates]

                    predictions[path] = active_labels

        # Ghi file output
        with open(output_file, "w", encoding="utf-8") as f:
            for path in image_paths:
                labels: List[str] = predictions[path]  # Dùng indexing [] thay cho .get()
                if labels:
                    f.write(f"{path},{','.join(labels)}\n")
                else:
                    f.write(f"{path}\n")

        print(f"Saved: {output_file}")


def main() -> None:

    # ============================================================================
    # GENERATE PERSON ATTRIBUTES PSEUDO LABELS
    # ============================================================================
    ckpt_path: str = (
        "models/checkpoints/fs26/person_attributes/classification_multilabel/"
        "v7.efficientnetv2m.cia+cia_gen+pa100k/weights/best-epoch=39-val_f1_macro=0.638.ckpt"
    )
    output_dir: str = "data/tmp/person_attributes_labels/pseudo"

    # # ==================== NHÓM A: person view data ====================
    # # Cắt bỏ phần data/tmp/singlelabel_to_multilabel/person_view/
    # # Tất cả các nhãn đều được phép lấy
    # group_a_files: List[str] = [
    #     "data/tmp/singlelabel_to_multilabel/person_view/back/satudora10k--train.txt",
    #     "data/tmp/singlelabel_to_multilabel/person_view/front/satudora10k--train.txt",
    #     "data/tmp/singlelabel_to_multilabel/person_view/side/satudora10k--train.txt",
    #     "data/tmp/singlelabel_to_multilabel/person_view/back/satudora10k--val.txt",
    #     "data/tmp/singlelabel_to_multilabel/person_view/front/satudora10k--val.txt",
    #     "data/tmp/singlelabel_to_multilabel/person_view/side/satudora10k--val.txt",
    # ]
    # group_a_prefix: str = "data/tmp/singlelabel_to_multilabel/person_view/"

    # print("--- Running Group A ---")
    # generate_pseudo_labels(
    #     ckpt_path=ckpt_path,
    #     file_list=group_a_files,
    #     output_dir=output_dir,
    #     prefix_to_strip=group_a_prefix,
    #     allowed_classes=None,
    #     threshold=THRESHOLD,
    #     device_name=DEVICE,
    # )

    # ==================== NHÓM B: action recognition data ====================
    # Cắt bỏ phần data/processed/fs26/action_recognition/classification_multilabel/
    # Tất cả các nhãn đều được phép lấy
    # (Bỏ comment phần bên dưới khi chạy tiếp Nhóm B)
    group_b_files: List[str] = [
        # "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar.txt",
        # "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_quan/train_resagepar.txt",
        # "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar_v2.txt",
        # "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_quan/train_resagepar_v2.txt",
        #
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.easy.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.medium.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.easy.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.medium.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.easy.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.medium.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.easy.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.medium.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.easy.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.medium.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.easy.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.medium.txt",
        #
        # gen-flux--set-1
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_gio_xe_hang/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_ke/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_quan/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_cam_san_pham/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_khong_cam_san_pham/train.txt",
        #
        # gen-flux--set-2..12
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-3/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-4/cho_tay_vao_tui_quan/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-5/cho_tay_vao_tui_quan/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-7/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-8/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-9/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-10/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-11/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-12/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        #
        # gen-sensenova--set-1..2
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
    ]
    group_b_prefix: str = (
        "data/tmp/action_recognition_labels/gt/"
    )
    
    print("\n--- Running Group B ---")
    generate_pseudo_labels(
        ckpt_path=ckpt_path,
        file_list=group_b_files,
        output_dir=output_dir,
        prefix_to_strip=group_b_prefix,
        allowed_classes=None,
        threshold=THRESHOLD,
        device_name=DEVICE,
    )

    # # ==================== NHÓM C ====================
    # # Cắt bỏ phần data/processed/fs26/person_attributes/classification_multilabel/
    # # Chỉ lấy các nhãn: basket, cart, product
    # # (Bỏ comment phần bên dưới khi chạy tiếp Nhóm C)
    # group_c_files: List[str] = [
    #     "data/processed/fs26/person_attributes/classification_multilabel/pa100k_train.txt",
    #     "data/processed/fs26/person_attributes/classification_multilabel/pa100k_val.txt",
    #     "data/processed/fs26/person_attributes/classification_multilabel/pa100k_test.txt",
    # ]
    # group_c_prefix: str = (
    #     "data/processed/fs26/person_attributes/classification_multilabel/"
    # )
    # group_c_labels: Set[str] = {"basket", "cart", "product"}
    
    # print("\n--- Running Group C ---")
    # generate_pseudo_labels(
    #     ckpt_path=ckpt_path,
    #     file_list=group_c_files,
    #     output_dir=output_dir,
    #     prefix_to_strip=group_c_prefix,
    #     allowed_classes=group_c_labels,
    #     threshold=THRESHOLD,
    #     device_name=DEVICE,
    # )


if __name__ == "__main__":
    main()
