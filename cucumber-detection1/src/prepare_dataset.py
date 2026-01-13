from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_utils import count_dataset_images, prepare_auto_train_dataset


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare YOLO train/val datasets from auto_train inputs."
    )
    parser.add_argument("--dataset", default="configs/dataset.yaml")
    parser.add_argument(
        "--images",
        default=str(project_root / "auto_train" / "cucumbers"),
        help="Path to cucumber images directory.",
    )
    parser.add_argument(
        "--negatives",
        default=str(project_root / "auto_train" / "not_cucumbers"),
        help="Path to non-cucumber images directory.",
    )
    parser.add_argument(
        "--labels",
        default=str(project_root / "auto_train" / "uncertain"),
        help="Path to labels directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--min-val", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_auto_train_dataset(
        dataset=args.dataset,
        cucumbers_dir=Path(args.images),
        negatives_dir=Path(args.negatives),
        labels_dir=Path(args.labels),
        val_ratio=args.val_ratio,
    )
    train_count, val_count = count_dataset_images(args.dataset)
    print(f"Prepared dataset: train={train_count}, val={val_count}")
    if train_count < args.min_train or val_count < args.min_val:
        print(
            "Warning: dataset is small for reliable training. "
            f"Recommended minimum is train={args.min_train}, val={args.min_val}."
        )


if __name__ == "__main__":
    main()
