from __future__ import annotations

import argparse
from pathlib import Path

from src.bootstrap_labels import bootstrap_labels
from src.train import train_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-train pipeline: pseudo-label images then train YOLO."
    )
    parser.add_argument(
        "--images",
        default=r"D:\ProjectPyCharm\cucumber-detection1\auto_train\cucumbers",
        help="Path to cucumber images directory.",
    )
    parser.add_argument(
        "--negatives",
        default=r"D:\ProjectPyCharm\cucumber-detection1\auto_train\not_cucumbers",
        help="Path to non-cucumber images directory.",
    )
    parser.add_argument(
        "--labels",
        default=r"D:\ProjectPyCharm\cucumber-detection1\auto_train\uncertain",
        help="Path to labels directory.",
    )
    parser.add_argument("--prior", default="configs/cucumber_prior.yaml")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--dataset", default="configs/dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="outputs/training")
    parser.add_argument("--name", default="auto")
    parser.add_argument("--device", default=None, help="cuda or cpu; auto if not set")
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--weights-dir", default="weights")
    return parser.parse_args()


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
    _create_empty_labels(Path(args.negatives), Path(args.labels))
    train_model(
        dataset=args.dataset,
        model_path=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        save_period=args.save_period,
        resume=args.resume,
        weights_dir=args.weights_dir,
    )


def _create_empty_labels(images_dir: Path, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            label_path.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
