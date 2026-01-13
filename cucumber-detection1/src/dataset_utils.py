from __future__ import annotations

import random
import shutil
from pathlib import Path

from .config import load_yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _resolve_base_path(dataset_path: Path) -> Path:
    data = load_yaml(dataset_path)
    base = data.get("path", "")
    base_path = Path(base)
    if base_path.is_absolute():
        return base_path

    resolved_from_config = (dataset_path.parent / base_path).resolve()
    project_root = dataset_path.parents[1] / base_path
    if resolved_from_config.exists() or not project_root.exists():
        return resolved_from_config
    return project_root.resolve()


def _resolve_images_dir(dataset_path: Path, split: str) -> Path | None:
    data = load_yaml(dataset_path)
    entry = data.get(split)
    if not entry or not isinstance(entry, str):
        return None
    entry_path = Path(entry)
    if entry_path.is_absolute():
        return entry_path
    return _resolve_base_path(dataset_path) / entry_path


def _labels_dir_for_images(images_dir: Path) -> Path:
    if images_dir.parent.name == "images":
        return images_dir.parent.parent / "labels" / images_dir.name
    return images_dir.parent / "labels"


def label_name_for_image(image_path: Path, root_dir: Path) -> str:
    try:
        relative = image_path.relative_to(root_dir)
    except ValueError:
        relative = image_path.name
    stem = Path(relative).with_suffix("").as_posix().replace("/", "__")
    return f"{root_dir.name}__{stem}"


def _collect_images(*dirs: Path) -> list[Path]:
    images: list[Path] = []
    for folder in dirs:
        if not folder.exists():
            continue
        for image_path in sorted(folder.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(image_path)
    return images


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for entry in path.glob("*"):
        if entry.is_file():
            entry.unlink()


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(
        1
        for image_path in directory.rglob("*")
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_dataset_images(dataset: str) -> tuple[int, int]:
    dataset_path = Path(dataset).expanduser().resolve()
    train_images_dir = _resolve_images_dir(dataset_path, "train")
    val_images_dir = _resolve_images_dir(dataset_path, "val")
    if train_images_dir is None or val_images_dir is None:
        return 0, 0
    return count_images(train_images_dir), count_images(val_images_dir)


def _root_for_image(image_path: Path, cucumbers_dir: Path, negatives_dir: Path) -> Path:
    if image_path.is_relative_to(cucumbers_dir):
        return cucumbers_dir
    if negatives_dir.exists() and image_path.is_relative_to(negatives_dir):
        return negatives_dir
    return image_path.parent


def prepare_auto_train_dataset(
    *,
    dataset: str,
    cucumbers_dir: Path,
    negatives_dir: Path,
    labels_dir: Path,
    val_ratio: float = 0.2,
) -> None:
    dataset_path = Path(dataset).expanduser().resolve()
    train_images_dir = _resolve_images_dir(dataset_path, "train")
    val_images_dir = _resolve_images_dir(dataset_path, "val")
    if train_images_dir is None or val_images_dir is None:
        return

    train_labels_dir = _labels_dir_for_images(train_images_dir)
    val_labels_dir = _labels_dir_for_images(val_images_dir)
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    _clear_directory(train_images_dir)
    _clear_directory(val_images_dir)
    _clear_directory(train_labels_dir)
    _clear_directory(val_labels_dir)

    images = _collect_images(cucumbers_dir, negatives_dir)
    if not images:
        return
    random.shuffle(images)

    total = len(images)
    val_count = int(total * val_ratio)
    if total > 1 and val_count == 0:
        val_count = 1
    split_index = total - val_count if val_count > 0 else total
    train_images = images[:split_index]
    val_images = images[split_index:] if val_count > 0 else []

    if not val_images:
        val_images = train_images

    for image_path in train_images:
        root_dir = _root_for_image(image_path, cucumbers_dir, negatives_dir)
        label_name = label_name_for_image(image_path, root_dir)
        target = train_images_dir / f"{label_name}{image_path.suffix.lower()}"
        shutil.copy2(image_path, target)
        label_source = labels_dir / f"{label_name}.txt"
        label_target = train_labels_dir / f"{label_name}.txt"
        if label_source.exists():
            shutil.copy2(label_source, label_target)
        elif not label_target.exists():
            label_target.write_text("", encoding="utf-8")

    for image_path in val_images:
        root_dir = _root_for_image(image_path, cucumbers_dir, negatives_dir)
        label_name = label_name_for_image(image_path, root_dir)
        target = val_images_dir / f"{label_name}{image_path.suffix.lower()}"
        shutil.copy2(image_path, target)
        label_source = labels_dir / f"{label_name}.txt"
        label_target = val_labels_dir / f"{label_name}.txt"
        if label_source.exists():
            shutil.copy2(label_source, label_target)
        elif not label_target.exists():
            label_target.write_text("", encoding="utf-8")
