from __future__ import annotations

from pathlib import Path

from src.bootstrap_labels import bootstrap_labels
from src.infer_image import infer_image
from src.infer_video import infer_video
from src.train import train_model


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def _ask_int(prompt: str, default: int) -> int:
    raw = _ask(prompt, str(default))
    return int(raw)


def _ask_float(prompt: str, default: float) -> float:
    raw = _ask(prompt, str(default))
    return float(raw)


def _print_menu() -> None:
    print("\n=== Меню управления проектом ===")
    print("1. Автообучение (псевдо-разметка + обучение)")
    print("2. Обучение модели")
    print("3. Псевдо-разметка изображений")
    print("4. Инференс на изображении")
    print("5. Инференс на видео/камере")
    print("0. Выход")


def run_auto_train() -> None:
    images = _ask("Папка с изображениями", "data/images")
    labels = _ask("Папка для меток", "data/labels")
    prior = _ask("Файл prior", "configs/cucumber_prior.yaml")
    class_id = _ask_int("ID класса", 0)
    dataset = _ask("Dataset yaml", "configs/dataset.yaml")
    model = _ask("Базовая модель", "yolov8n.pt")
    epochs = _ask_int("Эпохи", 50)
    imgsz = _ask_int("Размер изображения", 640)
    batch = _ask_int("Batch size", 16)
    project = _ask("Папка для обучения", "outputs/training")
    name = _ask("Название эксперимента", "auto")
    save_period = _ask_int("Период сохранения (эпохи)", 1)
    resume = _ask("Продолжить обучение? (y/n)", "n").lower().startswith("y")
    weights_dir = _ask("Папка для весов", "weights")

    bootstrap_labels(
        images_dir=Path(images),
        labels_dir=Path(labels),
        prior_path=prior,
        class_id=class_id,
    )
    train_model(
        dataset=dataset,
        model_path=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=None,
        save_period=save_period,
        resume=resume,
        weights_dir=weights_dir,
    )


def run_train() -> None:
    dataset = _ask("Dataset yaml", "configs/dataset.yaml")
    model = _ask("Базовая модель", "yolov8n.pt")
    epochs = _ask_int("Эпохи", 50)
    imgsz = _ask_int("Размер изображения", 640)
    batch = _ask_int("Batch size", 16)
    project = _ask("Папка для обучения", "outputs/training")
    name = _ask("Название эксперимента", "exp")
    save_period = _ask_int("Период сохранения (эпохи)", 1)
    resume = _ask("Продолжить обучение? (y/n)", "n").lower().startswith("y")
    weights_dir = _ask("Папка для весов", "weights")

    train_model(
        dataset=dataset,
        model_path=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=None,
        save_period=save_period,
        resume=resume,
        weights_dir=weights_dir,
    )


def run_bootstrap() -> None:
    images = _ask("Папка с изображениями", "data/images")
    labels = _ask("Папка для меток", "data/labels")
    prior = _ask("Файл prior", "configs/cucumber_prior.yaml")
    class_id = _ask_int("ID класса", 0)

    bootstrap_labels(
        images_dir=Path(images),
        labels_dir=Path(labels),
        prior_path=prior,
        class_id=class_id,
    )


def run_infer_image() -> None:
    weights = _ask("Весы модели", "weights/best.pt")
    image_path = _ask("Путь к изображению", "data/test.jpg")
    output = _ask("Путь для сохранения (пусто = авто)", "")
    conf = _ask_float("Confidence", 0.25)
    infer_image(
        weights=weights,
        image_path=image_path,
        output=output or None,
        conf=conf,
    )


def run_infer_video() -> None:
    weights = _ask("Весы модели", "weights/best.pt")
    source = _ask("Путь к видео или RTSP", "data/test.mp4")
    output = _ask("Путь для сохранения (пусто = не сохранять)", "")
    conf = _ask_float("Confidence", 0.25)
    display = _ask("Показывать окно? (y/n)", "y").lower().startswith("y")
    infer_video(
        weights=weights,
        source=source,
        conf=conf,
        display=display,
        output=output or None,
    )


def main() -> None:
    while True:
        _print_menu()
        choice = input("Выберите пункт: ").strip()
        if choice == "1":
            run_auto_train()
        elif choice == "2":
            run_train()
        elif choice == "3":
            run_bootstrap()
        elif choice == "4":
            run_infer_image()
        elif choice == "5":
            run_infer_video()
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неизвестный пункт меню.")


if __name__ == "__main__":
    main()
