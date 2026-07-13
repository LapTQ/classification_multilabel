import os
from typing import Dict, List, Set


def filter_annotation_files(
    input_files: List[str],
    desired_labels: List[str],
    output_file: str,
) -> None:
    """Filters labels in annotation files, keeping only the desired labels.

    Maintains the exact list of images, writing empty labels if none of the
    desired labels match.

    Args:
        input_files: List of paths to input annotation text files.
        desired_labels: List of label names to keep.
        output_file: Path to the output text file.
    """
    desired_set: Set[str] = {lbl.strip().lower() for lbl in desired_labels}
    casing_map: Dict[str, str] = {
        lbl.strip().lower(): lbl.strip() for lbl in desired_labels
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    line_count: int = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for filepath in input_files:
            if not os.path.exists(filepath):
                print(f"Warning: Input file '{filepath}' does not exist.")
                continue

            with open(filepath, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue

                    parts: List[str] = line.split(",")
                    img_path: str = parts[0].strip()
                    labels: List[str] = parts[1:]

                    filtered_labels: List[str] = []
                    for lbl in labels:
                        lbl_clean: str = lbl.strip().lower()
                        if lbl_clean in desired_set:
                            filtered_labels.append(casing_map[lbl_clean])

                    new_line: str = img_path + "," + ",".join(filtered_labels)
                    out_f.write(new_line + "\n")
                    line_count += 1

    print(
        f"Filtered annotation file saved to: {output_file} (Total: {line_count} images)"
    )


def main() -> None:
    """Main execution function to demonstrate the label filtering script."""
    input_files: List[str] = [
        "/home/laptq/data/fs26/processed/train_data/classify_rgb_multilabel/person_attributes/Satudora_test.txt"
    ]
    desired_labels: List[str] = ["backpack", "shoulderbag", "handbag", "product"]
    output_file: str = "/home/laptq/classification_multilabel/outputs/trivials/filtered_Satudora_test.txt"

    print(f"Filtering files: {input_files}")
    print(f"Keeping labels: {desired_labels}")
    filter_annotation_files(input_files, desired_labels, output_file)


if __name__ == "__main__":
    main()
