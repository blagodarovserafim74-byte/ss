from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_yaml
from src.heuristics import draw_detections, find_cucumber_candidates, load_prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-label cucumber candidates from images.")
    parser.add_argument(
        "--images",
        default=r"D:\ProjectPyCharm\cucumber-detection1\auto_train\cucumbers",
        help="Path to cucumber images directory.",
    )
    parser.add_argument(
        "--labels",
        default=r"D:\ProjectPyCharm\cucumber-detection1\auto_train\uncertain",
        help="Output labels directory.",
    )
    parser.add_argument(
        "--prior",
        default="configs/cucumber_prior.yaml",
        help="Path to cucumber prior yaml.",
    )
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument(
        "--review",
        action="store_true",
        help="Show preview window and allow moderation with keyboard.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=0,
        help="Delay between frames in review mode (0 = wait for key).",
    )
    return parser.parse_args()


def bootstrap_labels(
    *,
    images_dir: Path,
    labels_dir: Path,
    prior_path: str,
    class_id: int,
    review: bool = False,
    delay_ms: int = 0,
) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    prior = load_prior(load_yaml(prior_path))

    window_name = "Псевдо-разметка: огурцы"
    try:
        for image_path in sorted(images_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            detections = find_cucumber_candidates(image, prior)

            if review:
                preview = draw_detections(image, detections)
                cv2.putText(
                    preview,
                    "Y - принять, N - отклонить, Q - выход",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    "Y - принять, N - отклонить, Q - выход",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview)
                wait = 0 if delay_ms <= 0 else delay_ms
                key = cv2.waitKey(wait) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("n"), ord("N")):
                    detections = []

            label_path = labels_dir / f"{image_path.stem}.txt"
            with label_path.open("w", encoding="utf-8") as handle:
                for det in detections:
                    x_center, y_center, box_w, box_h = det.to_yolo(width, height)
                    handle.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
                    )
    finally:
        if review:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    bootstrap_labels(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        prior_path=args.prior,
        class_id=args.class_id,
        review=args.review,
        delay_ms=args.delay_ms,
    )


if __name__ == "__main__":
    main()
