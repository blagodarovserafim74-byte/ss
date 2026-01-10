from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for cucumber detection.")
    parser.add_argument("--dataset", default="configs/dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="outputs/training")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--device", default=None, help="cuda or cpu; auto if not set")
    parser.add_argument("--save-period", type=int, default=1, help="How often to save checkpoints.")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint.")
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="Directory to copy best/last weights after training.",
    )
    return parser.parse_args()


def _copy_weights(save_dir: Path, weights_dir: Path) -> None:
    weights_source = save_dir / "weights"
    if not weights_source.exists():
        return

    weights_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "last.pt"):
        source = weights_source / name
        if source.exists():
            target = weights_dir / f"{save_dir.name}_{name}"
            shutil.copy2(source, target)
            print(f"Saved {name} to {target}")


def train_model(
    *,
    dataset: str,
    model_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: str,
    name: str,
    device: str | None,
    save_period: int,
    resume: bool,
    weights_dir: str,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    model.train(
        data=str(Path(dataset)),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        save=True,
        save_period=save_period,
        resume=resume,
        exist_ok=True,
    )

    save_dir = None
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        save_dir = Path(trainer.save_dir)
        _copy_weights(save_dir, Path(weights_dir))
    return save_dir if save_dir is not None else Path(project) / name


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
