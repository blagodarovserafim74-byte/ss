from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
import threading
import time

import torch
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config import load_yaml  # type: ignore[reportMissingImports]
else:
    from .config import load_yaml


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


def _autosave_loop(
    *,
    save_dir: Path,
    weights_dir: Path,
    interval_minutes: int,
    stop_event: threading.Event,
) -> None:
    interval_seconds = max(interval_minutes, 1) * 60
    weights_dir.mkdir(parents=True, exist_ok=True)
    last_seen_mtime: float | None = None
    while not stop_event.wait(interval_seconds):
        last_path = save_dir / "weights" / "last.pt"
        if not last_path.exists():
            continue
        try:
            mtime = last_path.stat().st_mtime
            if last_seen_mtime is None or mtime != last_seen_mtime:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                target = weights_dir / f"{save_dir.name}_autosave_{timestamp}.pt"
                shutil.copy2(last_path, target)
                print(f"Auto-saved checkpoint to {target}")
                last_seen_mtime = mtime
        except Exception as exc:
            print(f"Auto-save failed: {exc}")


def _resolve_dataset_paths(dataset_path: Path) -> list[Path]:
    data = load_yaml(dataset_path)
    base = data.get("path", "")
    base_path = Path(base)
    if not base_path.is_absolute():
        resolved_from_config = (dataset_path.parent / base_path).resolve()
        project_root = dataset_path.parents[1] / base_path
        if resolved_from_config.exists() or not project_root.exists():
            base_path = resolved_from_config
        else:
            base_path = project_root.resolve()

    resolved: list[Path] = []
    for split in ("train", "val", "test"):
        value = data.get(split)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            entries = value
        else:
            entries = [value]
        for entry in entries:
            entry_path = Path(entry)
            if not entry_path.is_absolute():
                entry_path = base_path / entry_path
            resolved.append(entry_path)
    return resolved


def _ensure_dataset_paths_exist(dataset: str) -> None:
    dataset_path = Path(dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_path}")

    missing = [path for path in _resolve_dataset_paths(dataset_path) if not path.exists()]
    if missing:
        project_root = dataset_path.parents[1]
        auto_images = project_root / "auto_train" / "cucumbers"
        auto_negatives = project_root / "auto_train" / "not_cucumbers"
        auto_labels = project_root / "auto_train" / "uncertain"
        if auto_images.exists() and auto_labels.exists():
            from .dataset_utils import prepare_auto_train_dataset

            prepare_auto_train_dataset(
                dataset=str(dataset_path),
                cucumbers_dir=auto_images,
                negatives_dir=auto_negatives,
                labels_dir=auto_labels,
            )
            missing = [path for path in _resolve_dataset_paths(dataset_path) if not path.exists()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        message = (
            "Dataset images not found. Please update the dataset paths or place data in the expected "
            f"location.\nDataset config: {dataset_path}\nMissing paths:\n{missing_list}"
        )
        raise FileNotFoundError(message)


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
    autosave_minutes: int | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dataset_paths_exist(dataset)
    model = YOLO(model_path)
    save_dir = Path(project) / name
    stop_event = threading.Event()
    autosave_thread: threading.Thread | None = None
    if autosave_minutes is not None and autosave_minutes > 0:
        autosave_thread = threading.Thread(
            target=_autosave_loop,
            kwargs={
                "save_dir": save_dir,
                "weights_dir": Path(weights_dir),
                "interval_minutes": autosave_minutes,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        autosave_thread.start()
    try:
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
    except Exception as exc:
        message = str(exc).lower()
        if resume and "nothing to resume" in message:
            print("Resume requested but no checkpoint to resume; starting fresh run.")
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
                resume=False,
                exist_ok=True,
            )
        else:
            raise

    stop_event.set()
    if autosave_thread is not None:
        autosave_thread.join(timeout=5)

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
