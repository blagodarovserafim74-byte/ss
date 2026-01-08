from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.config import load_yaml
from src.heuristics import find_cucumber_candidates, load_prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-label cucumber candidates from images.")
    parser.add_argument("--images", required=True, help="Path to images directory.")
    parser.add_argument("--labels", required=True, help="Output labels directory.")
    parser.add_argument(
        "--prior",
        default="configs/cucumber_prior.yaml",
        help="Path to cucumber prior yaml.",
    )
    parser.add_argument("--class-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)

    prior = load_prior(load_yaml(args.prior))

    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        detections = find_cucumber_candidates(image, prior)

        label_path = labels_dir / f"{image_path.stem}.txt"
        with label_path.open("w", encoding="utf-8") as handle:
            for det in detections:
                x_center, y_center, box_w, box_h = det.to_yolo(width, height)
                handle.write(
                    f"{args.class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
                )


if __name__ == "__main__":
    main()
