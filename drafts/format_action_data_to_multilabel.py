import os
from typing import Dict, List

import yaml


def map_input_path(original_path: str) -> str:
    """Maps the configuration paths to the actual filesystem paths."""
    # Replace the shoplifting-detection outputs path with classification outputs path
    prefix_to_replace = (
        "/home/laptq/laptq-fs26-shoplifting-detection/outputs/train_data/"
    )
    actual_prefix = "/home/laptq/classification/outputs/train_data/"
    if original_path.startswith(prefix_to_replace):
        return original_path.replace(prefix_to_replace, actual_prefix)
    return original_path


def format_txt_file(input_file: str, output_file: str, class_name: str) -> None:
    """Reads a single-label txt file, appends the class name to each line, and writes to output_file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    new_lines: List[str] = []
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            new_lines.append(f"{cleaned_line},{class_name}\n")

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.writelines(new_lines)


def main() -> None:
    """Main processing function to convert config and label files."""
    config_path = "/home/laptq/classification_multilabel/externals/classification/configs/v4.2dcnn.cluster-CNN-8--cut-l4.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    classes: List[str] = config["classes"]
    train_data_dict: Dict[str, List[str]] = config["train_data"]
    val_data_dict: Dict[str, List[str]] = config["val_data"]

    output_root = "/home/laptq/data/fs26/processed/train_data/classify_rgb_multilabel/labels_actions--cluster-CNN-8--cut-l4"

    # Process training data
    for class_name in classes:
        files = train_data_dict[class_name]
        for f_path in files:
            actual_input_path = map_input_path(f_path)
            # Determine relative path structure
            # e.g. cho_tay_vao_tui_quan/train--min4k--max5k.txt
            rel_path = os.path.join(class_name, os.path.basename(actual_input_path))
            dest_path = os.path.join(output_root, rel_path)
            print(f"Formatting Train: {actual_input_path} -> {dest_path}")
            format_txt_file(actual_input_path, dest_path, class_name)

    # Process validation data
    for class_name in classes:
        files = val_data_dict[class_name]
        for f_path in files:
            actual_input_path = map_input_path(f_path)
            rel_path = os.path.join(class_name, os.path.basename(actual_input_path))
            dest_path = os.path.join(output_root, rel_path)
            print(f"Formatting Val: {actual_input_path} -> {dest_path}")
            format_txt_file(actual_input_path, dest_path, class_name)


if __name__ == "__main__":
    main()
