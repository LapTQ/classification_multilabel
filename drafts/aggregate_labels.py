import os
from collections import defaultdict



def aggregate_group(
    source_files: list[str],
    output_file: str,
    merge_classes: dict[str, list[str]] | None = None,
) -> None:
    """Aggregates label files in the list, merging unique labels for each image."""
    label_to_target: dict[str, str] = {}
    if merge_classes is not None:
        for target_label, old_labels in merge_classes.items():
            for old_label in old_labels:
                label_to_target[old_label] = target_label

    ordered_images: list[str] = []
    image_to_labels: dict[str, set[str]] = defaultdict(set)

    for source_file in source_files:
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts: list[str] = line.split(",")
                img_path: str = parts[0]
                labels: list[str] = parts[1:]

                # Check if we have seen this image path before
                if img_path not in image_to_labels:
                    ordered_images.append(img_path)

                for label in labels:
                    label = label.strip()
                    if label:
                        mapped_label: str = (
                            label_to_target[label] if label in label_to_target else label
                        )
                        image_to_labels[img_path].add(mapped_label)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write merged labels to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for img_path in ordered_images:
            labels_set: set[str] = image_to_labels[img_path]
            if labels_set:
                sorted_labels: list[str] = sorted(list(labels_set))
                f.write(f"{img_path},{','.join(sorted_labels)}\n")
            else:
                f.write(f"{img_path}\n")

    print(f"Saved aggregated labels to: {output_file}")


def main() -> None:
    output_dir: str = "data/processed/fs26/action+attributes+view"

    # Cấu hình gộp nhãn: "class_gộp_mới": ["class_cũ_1", "class_cũ_2", ...]
    merge_classes: dict[str, list[str]] = {
        "tay_cam_san_pham": ["product", "tay_cam_san_pham"],
    }

    # Define target files and their source files based on d1.txt L113-L212
    groups: dict[str, list[str]] = {
        # ====================================================================
        # action.for_CNN.8_classes_grouped_123.manually_selected - train
        # ====================================================================
        "action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/train.txt",
        ],

        # ====================================================================
        # action.for_CNN.8_classes_grouped_123.manually_selected - val.easy
        # ====================================================================
        "action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.easy.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.easy.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.easy.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.easy.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.easy.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.easy.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.easy.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.easy.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.easy.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.easy.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.easy.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.easy.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.easy.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.easy.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.easy.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.easy.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.easy.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.easy.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.easy.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.easy.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.easy.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.easy.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.easy.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.easy.txt",
        ],

        # ====================================================================
        # action.for_CNN.8_classes_grouped_123.manually_selected - val.medium
        # ====================================================================
        "action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.medium.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.medium.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.medium.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_khong_cam_san_pham/val.medium.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.medium.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.medium.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.medium.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/val.medium.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.medium.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.medium.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.medium.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_ke/val.medium.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.medium.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.medium.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.medium.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_gio_xe_hang/val.medium.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.medium.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.medium.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.medium.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/tay_cam_san_pham/val.medium.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.medium.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.medium.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.medium.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.manually_selected/cho_tay_vao_tui_quan/val.medium.txt",
        ],

        # ====================================================================
        # action.for_CNN.8_classes_grouped_123.gen-flux--set-1
        # ====================================================================
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_khong_cam_san_pham/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_khong_cam_san_pham/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_khong_cam_san_pham/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_khong_cam_san_pham/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_ke/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_ke/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_ke/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_ke/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_gio_xe_hang/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_gio_xe_hang/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_gio_xe_hang/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_gio_xe_hang/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_cam_san_pham/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_cam_san_pham/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_cam_san_pham/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/tay_cam_san_pham/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_quan/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-1/cho_tay_vao_tui_quan/train.txt",
        ],

        # ====================================================================
        # action.for_CNN.8_classes_grouped_123.gen-flux--set-2..12
        # ====================================================================
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-3/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-3/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-3/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-3/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-4/cho_tay_vao_tui_quan/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-4/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-4/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-4/cho_tay_vao_tui_quan/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-5/cho_tay_vao_tui_quan/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-5/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-5/cho_tay_vao_tui_quan/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-5/cho_tay_vao_tui_quan/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-7/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-7/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-7/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-7/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-8/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-8/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-8/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-8/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-9/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-9/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-9/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-9/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-10/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-10/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-10/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-10/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-11/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-11/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-11/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-11/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-flux--set-12/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-flux--set-12/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-12/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-flux--set-12/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],

        # ====================================================================
        # action.for_CNN.8_classes_grouped_123.gen-sensenova--set-1..2
        # ====================================================================
        "action.for_CNN.8_classes_grouped_123.gen-sensenova--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-1/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        "action.for_CNN.8_classes_grouped_123.gen-sensenova--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt": [
            "data/tmp/action_recognition_labels/gt/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_view_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
            "data/tmp/person_attributes_labels/pseudo/action.for_CNN.8_classes_grouped_123.gen-sensenova--set-2/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train.txt",
        ],
        # "train_resagepar.txt": [
        #     "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar.txt",
        #     "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_quan/train_resagepar.txt",
        #     "data/tmp/person_attributes_labels/pseudo/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar.txt",
        #     "data/tmp/person_attributes_labels/pseudo/resagepar/cho_tay_vao_tui_quan/train_resagepar.txt",
        #     "data/tmp/person_view_labels/pseudo/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar.txt",
        #     "data/tmp/person_view_labels/pseudo/resagepar/cho_tay_vao_tui_quan/train_resagepar.txt",
        # ],
        # "train_resagepar_v2.txt": [
        #     "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar_v2.txt",
        #     "data/tmp/action_recognition_labels/gt/resagepar/cho_tay_vao_tui_quan/train_resagepar_v2.txt",
        #     "data/tmp/person_attributes_labels/pseudo/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar_v2.txt",
        #     "data/tmp/person_attributes_labels/pseudo/resagepar/cho_tay_vao_tui_quan/train_resagepar_v2.txt",
        #     "data/tmp/person_view_labels/pseudo/resagepar/cho_tay_vao_tui_ao_tui_deo_tui_cam_tay/train_resagepar_v2.txt",
        #     "data/tmp/person_view_labels/pseudo/resagepar/cho_tay_vao_tui_quan/train_resagepar_v2.txt",
        # ],
        "satudora10k--train.txt": [
            "data/tmp/action_recognition_labels/pseudo/front/satudora10k--train.txt",
            "data/tmp/action_recognition_labels/pseudo/back/satudora10k--train.txt",
            "data/tmp/action_recognition_labels/pseudo/side/satudora10k--train.txt",
            "data/tmp/person_view_labels/gt/front/satudora10k--train.txt",
            "data/tmp/person_view_labels/gt/back/satudora10k--train.txt",
            "data/tmp/person_view_labels/gt/side/satudora10k--train.txt",
            "data/tmp/person_attributes_labels/pseudo/front/satudora10k--train.txt",
            "data/tmp/person_attributes_labels/pseudo/back/satudora10k--train.txt",
            "data/tmp/person_attributes_labels/pseudo/side/satudora10k--train.txt",
        ],
        "satudora10k--val.txt": [
            "data/tmp/action_recognition_labels/pseudo/front/satudora10k--val.txt",
            "data/tmp/action_recognition_labels/pseudo/back/satudora10k--val.txt",
            "data/tmp/action_recognition_labels/pseudo/side/satudora10k--val.txt",
            "data/tmp/person_view_labels/gt/front/satudora10k--val.txt",
            "data/tmp/person_view_labels/gt/back/satudora10k--val.txt",
            "data/tmp/person_view_labels/gt/side/satudora10k--val.txt",
            "data/tmp/person_attributes_labels/pseudo/front/satudora10k--val.txt",
            "data/tmp/person_attributes_labels/pseudo/back/satudora10k--val.txt",
            "data/tmp/person_attributes_labels/pseudo/side/satudora10k--val.txt",
        ],
        "Satudora_test.txt": [
            "data/tmp/action_recognition_labels/pseudo/Satudora_test.txt",
            "data/tmp/person_view_labels/pseudo/Satudora_test.txt",
            "data/tmp/person_attributes_labels/gt/Satudora_test.txt",
        ],
        "pa100k_train.txt": [
            "data/tmp/action_recognition_labels/pseudo/pa100k_train.txt",
            "data/tmp/person_view_labels/gt/front/pa100k--train.txt",
            "data/tmp/person_view_labels/gt/back/pa100k--train.txt",
            "data/tmp/person_view_labels/gt/side/pa100k--train.txt",
            "data/tmp/person_attributes_labels/gt/pa100k_train.txt",
            "data/tmp/person_attributes_labels/pseudo/pa100k_train.txt",
        ],
        "pa100k_val.txt": [
            "data/tmp/action_recognition_labels/pseudo/pa100k_val.txt",
            "data/tmp/person_view_labels/gt/front/pa100k--val.txt",
            "data/tmp/person_view_labels/gt/back/pa100k--val.txt",
            "data/tmp/person_view_labels/gt/side/pa100k--val.txt",
            "data/tmp/person_attributes_labels/gt/pa100k_val.txt",
            "data/tmp/person_attributes_labels/pseudo/pa100k_val.txt",
        ],
        "pa100k_test.txt": [
            "data/tmp/action_recognition_labels/pseudo/pa100k_test.txt",
            "data/tmp/person_view_labels/gt/front/pa100k--test.txt",
            "data/tmp/person_view_labels/gt/back/pa100k--test.txt",
            "data/tmp/person_view_labels/gt/side/pa100k--test.txt",
            "data/tmp/person_attributes_labels/gt/pa100k_test.txt",
            "data/tmp/person_attributes_labels/pseudo/pa100k_test.txt",
        ],
        "CIA_original.txt": [
            "data/tmp/action_recognition_labels/pseudo/CIA_original.txt",
            "data/tmp/person_view_labels/pseudo/CIA_original.txt",
            "data/tmp/person_attributes_labels/gt/CIA_original.txt",
        ],
        "CIA_combined_with_geneated_editting.txt": [
            "data/tmp/action_recognition_labels/pseudo/CIA_combined_with_geneated_editting.txt",
            "data/tmp/person_view_labels/pseudo/CIA_combined_with_geneated_editting.txt",
            "data/tmp/person_attributes_labels/gt/CIA_combined_with_geneated_editting.txt",
        ],
    }

    # Aggregate each group
    for target_name, source_files in groups.items():
        output_file: str = os.path.join(output_dir, target_name)
        aggregate_group(source_files, output_file, merge_classes=merge_classes)


if __name__ == "__main__":
    main()
