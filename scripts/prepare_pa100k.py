import os
from typing import Dict, List
import numpy as np
import scipy.io as sio


def process_split(
    image_names: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    image_dir: str,
    attribute_indices: Dict[int, str],
) -> None:
    """Processes a split (train, val, test) and writes the annotations to a text file.

    Args:
        image_names: Array of image names.
        labels: Array of labels.
        output_path: Path to the output text file.
        image_dir: Directory containing the images.
        attribute_indices: Dictionary mapping label indices to attribute names.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in range(len(image_names)):
            img_name: str = str(image_names[idx][0][0])
            img_path: str = os.path.join(image_dir, img_name)

            # Find which interested attributes are active (value == 1)
            row_labels: np.ndarray = labels[idx]
            active_attributes: List[str] = []
            for attr_idx, attr_name in attribute_indices.items():
                if row_labels[attr_idx] == 1:
                    active_attributes.append(attr_name)

            # Format: img_path,label1,label2,...
            # If no active attributes, it should end with a comma (e.g. img_path,)
            line: str = img_path + "," + ",".join(active_attributes)
            f.write(line + "\n")


def main() -> None:
    """Main execution function to load the mat file and write output splits."""
    mat_path: str = "/home/laptq/classification_multilabel/data/person-attributes/pa100k/annotation/annotation.mat"
    image_dir: str = "/home/laptq/classification_multilabel/data/person-attributes/pa100k/images"
    output_dir: str = "/home/laptq/classification_multilabel/outputs/train_data/classify_rgb_multilabel"

    # Interested attributes and their corresponding indices in the 26-dim label vector
    attribute_indices: Dict[int, str] = {
        4: "front",
        5: "side",
        6: "back",
        9: "handbag",
        10: "shoulderbag",
        11: "backpack",
        12: "holdobjectsinfront",
    }

    print(f"Loading annotation mat file from: {mat_path}")
    mat: Dict = sio.loadmat(mat_path)

    # Verify and print attribute list from mat file to show match
    print("\n--- Attribute Indices ---")
    all_attributes = mat["attributes"]
    for i, attr in enumerate(all_attributes):
        attr_name: str = str(attr[0][0])
        print(f"  Index {i:2d}: {attr_name}")
    print("-----------------------------------------\n")

    splits: List[Dict] = [
        {
            "name": "train",
            "images": mat["train_images_name"],
            "labels": mat["train_label"],
            "output": os.path.join(output_dir, "pa100k_train.txt"),
        },
        {
            "name": "val",
            "images": mat["val_images_name"],
            "labels": mat["val_label"],
            "output": os.path.join(output_dir, "pa100k_val.txt"),
        },
        {
            "name": "test",
            "images": mat["test_images_name"],
            "labels": mat["test_label"],
            "output": os.path.join(output_dir, "pa100k_test.txt"),
        },
    ]

    for split in splits:
        split_name: str = split["name"]
        print(f"Processing split: {split_name}")
        process_split(
            image_names=split["images"],
            labels=split["labels"],
            output_path=split["output"],
            image_dir=image_dir,
            attribute_indices=attribute_indices,
        )
        print(f"Finished processing split: {split_name}, saved to: {split['output']}")


if __name__ == "__main__":
    main()
