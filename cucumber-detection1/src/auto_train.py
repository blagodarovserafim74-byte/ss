from __future__ import annotations

import argparse
from pathlib import Path

from .bootstrap_labels import bootstrap_labels
from .dataset_utils import label_name_for_image, prepare_auto_train_dataset
from .train import train_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Auto-train pipeline: pseudo-label images then train YOLO."
    )
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
    parser.add_argument("--prior", default="configs/cucumber_prior.yaml")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument(
        "--unattended",
        action="store_true",
        help="Run without preview windows or user moderation.",
    )
    parser.add_argument(
        "--autosave-minutes",
        type=int,
        default=15,
        help="Auto-save checkpoint interval in minutes when unattended.",
    )
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
    review = args.review if not args.unattended else False
    delay_ms = args.delay_ms if not args.unattended else 0
    autosave_minutes = args.autosave_minutes if args.unattended else None
    bootstrap_labels(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        prior_path=args.prior,
        class_id=args.class_id,
        review=review,
        delay_ms=delay_ms,
    )
    _create_empty_labels(Path(args.negatives), Path(args.labels))
    prepare_auto_train_dataset(
        dataset=args.dataset,
        cucumbers_dir=Path(args.images),
        negatives_dir=Path(args.negatives),
        labels_dir=Path(args.labels),
    )
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
        autosave_minutes=autosave_minutes,
    )


def _create_empty_labels(images_dir: Path, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_name = label_name_for_image(image_path, images_dir)
        label_path = labels_dir / f"{label_name}.txt"
        if not label_path.exists():
            label_path.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
