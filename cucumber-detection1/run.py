from __future__ import annotations

import importlib
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR
if not (PROJECT_DIR / "src").exists() and (ROOT_DIR / "cucumber-detection1" / "src").exists():
    PROJECT_DIR = ROOT_DIR / "cucumber-detection1"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

bootstrap_labels = importlib.import_module("src.bootstrap_labels").bootstrap_labels
infer_image = importlib.import_module("src.infer_image").infer_image
infer_video = importlib.import_module("src.infer_video").infer_video
train_model = importlib.import_module("src.train").train_model
get_logger = importlib.import_module("src.logging_utils").get_logger


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LOGGER = get_logger("run", Path("logs"))


def _create_empty_labels(images_dir: Path, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            label_path.write_text("", encoding="utf-8")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Огурцы: обучение и инференс")
        self.geometry("880x700")
        self.resizable(False, False)
        self.configure(bg="#f4f6f8")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background="#f4f6f8")
        style.configure("TFrame", background="#f4f6f8")
        style.configure("TLabel", background="#f4f6f8", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        notebook.add(self._build_auto_tab(notebook), text="Автообучение")
        notebook.add(self._build_train_tab(notebook), text="Обучение")
        notebook.add(self._build_bootstrap_tab(notebook), text="Псевдо-разметка")
        notebook.add(self._build_infer_image_tab(notebook), text="Инференс: Фото")
        notebook.add(self._build_infer_video_tab(notebook), text="Инференс: Видео/Камера")

    @staticmethod
    def _row(parent: ttk.Frame, label: str, default: str, row: int) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=8)
        entry = ttk.Entry(parent, width=70)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky="w", pady=4, padx=8)
        return entry

    def _build_auto_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(
            frame,
            text="Автообучение: разметка → обучение",
            style="Header.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(6, 10), padx=8)
        images = self._row(
            frame,
            "Папка с огурцами",
            r"D:\ProjectPyCharm\cucumber-detection1\auto_train\cucumbers",
            1,
        )
        negatives = self._row(
            frame,
            "Папка с не-огурцами",
            r"D:\ProjectPyCharm\cucumber-detection1\auto_train\not_cucumbers",
            2,
        )
        labels = self._row(
            frame,
            "Папка для меток",
            r"D:\ProjectPyCharm\cucumber-detection1\auto_train\uncertain",
            3,
        )
        prior = self._row(frame, "Файл prior", "configs/cucumber_prior.yaml", 4)
        class_id = self._row(frame, "ID класса", "0", 5)
        dataset = self._row(frame, "Dataset yaml", "configs/dataset.yaml", 6)
        model = self._row(frame, "Базовая модель", "yolov8n.pt", 7)
        epochs = self._row(frame, "Эпохи", "50", 8)
        imgsz = self._row(frame, "Размер изображения", "640", 9)
        batch = self._row(frame, "Batch size", "16", 10)
        project = self._row(frame, "Папка для обучения", "outputs/training", 11)
        name = self._row(frame, "Название эксперимента", "auto", 12)
        save_period = self._row(frame, "Период сохранения (эпохи)", "1", 13)
        resume = self._row(frame, "Продолжить обучение? (y/n)", "n", 14)
        weights_dir = self._row(frame, "Папка для весов", "weights", 15)
        review = self._row(frame, "Модерация? (y/n)", "y", 16)
        delay_ms = self._row(frame, "Задержка (мс, 0=ожидать)", "0", 17)

        def run_auto() -> None:
            try:
                bootstrap_labels(
                    images_dir=Path(images.get()),
                    labels_dir=Path(labels.get()),
                    prior_path=prior.get(),
                    class_id=int(class_id.get()),
                    review=review.get().lower().startswith("y"),
                    delay_ms=int(delay_ms.get()),
                )
                _create_empty_labels(Path(negatives.get()), Path(labels.get()))
                train_model(
                    dataset=dataset.get(),
                    model_path=model.get(),
                    epochs=int(epochs.get()),
                    imgsz=int(imgsz.get()),
                    batch=int(batch.get()),
                    project=project.get(),
                    name=name.get(),
                    device=None,
                    save_period=int(save_period.get()),
                    resume=resume.get().lower().startswith("y"),
                    weights_dir=weights_dir.get(),
                )
                messagebox.showinfo("Готово", "Автообучение завершено.")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

        ttk.Button(frame, text="Запустить автообучение", command=run_auto).grid(
            row=18, column=0, columnspan=2, pady=14
        )
        return frame

    def _build_train_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Ручное обучение", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(6, 10), padx=8
        )
        dataset = self._row(frame, "Dataset yaml", "configs/dataset.yaml", 1)
        model = self._row(frame, "Базовая модель", "yolov8n.pt", 2)
        epochs = self._row(frame, "Эпохи", "50", 3)
        imgsz = self._row(frame, "Размер изображения", "640", 4)
        batch = self._row(frame, "Batch size", "16", 5)
        project = self._row(frame, "Папка для обучения", "outputs/training", 6)
        name = self._row(frame, "Название эксперимента", "exp", 7)
        save_period = self._row(frame, "Период сохранения (эпохи)", "1", 8)
        resume = self._row(frame, "Продолжить обучение? (y/n)", "n", 9)
        weights_dir = self._row(frame, "Папка для весов", "weights", 10)

        def run_train() -> None:
            try:
                train_model(
                    dataset=dataset.get(),
                    model_path=model.get(),
                    epochs=int(epochs.get()),
                    imgsz=int(imgsz.get()),
                    batch=int(batch.get()),
                    project=project.get(),
                    name=name.get(),
                    device=None,
                    save_period=int(save_period.get()),
                    resume=resume.get().lower().startswith("y"),
                    weights_dir=weights_dir.get(),
                )
                messagebox.showinfo("Готово", "Обучение завершено.")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

        ttk.Button(frame, text="Запустить обучение", command=run_train).grid(
            row=11, column=0, columnspan=2, pady=14
        )
        return frame

    def _build_bootstrap_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Псевдо-разметка", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(6, 10), padx=8
        )
        images = self._row(
            frame,
            "Папка с огурцами",
            r"D:\ProjectPyCharm\cucumber-detection1\auto_train\cucumbers",
            1,
        )
        labels = self._row(
            frame,
            "Папка для меток",
            r"D:\ProjectPyCharm\cucumber-detection1\auto_train\uncertain",
            2,
        )
        prior = self._row(frame, "Файл prior", "configs/cucumber_prior.yaml", 3)
        class_id = self._row(frame, "ID класса", "0", 4)
        review = self._row(frame, "Модерация? (y/n)", "y", 5)
        delay_ms = self._row(frame, "Задержка (мс, 0=ожидать)", "0", 6)

        def run_bootstrap() -> None:
            try:
                bootstrap_labels(
                    images_dir=Path(images.get()),
                    labels_dir=Path(labels.get()),
                    prior_path=prior.get(),
                    class_id=int(class_id.get()),
                    review=review.get().lower().startswith("y"),
                    delay_ms=int(delay_ms.get()),
                )
                messagebox.showinfo("Готово", "Псевдо-разметка завершена.")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

        ttk.Button(frame, text="Запустить псевдо-разметку", command=run_bootstrap).grid(
            row=7, column=0, columnspan=2, pady=14
        )
        return frame

    def _build_infer_image_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Инференс по фото", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(6, 10), padx=8
        )
        weights = self._row(frame, "Весы модели", "weights/best.pt", 1)
        image_path = self._row(frame, "Путь к изображению", "data/test.jpg", 2)
        output = self._row(frame, "Путь для сохранения (пусто = авто)", "", 3)
        conf = self._row(frame, "Confidence", "0.25", 4)
        display = self._row(frame, "Показывать окно? (y/n)", "y", 5)
        delay_ms = self._row(frame, "Задержка (мс, 0=ожидать)", "0", 6)

        def run_infer() -> None:
            try:
                _, detected = infer_image(
                    weights=weights.get(),
                    image_path=image_path.get(),
                    output=output.get() or None,
                    conf=float(conf.get()),
                    display=display.get().lower().startswith("y"),
                    delay_ms=int(delay_ms.get()),
                )
                message = "Огурец найден." if detected else "Огурец не найден."
                output_path = output.get() or Path(image_path.get()).with_suffix(
                    ".detected.jpg"
                )
                LOGGER.info("Image inference completed: %s", message)
                messagebox.showinfo(
                    "Готово",
                    f"Инференс завершен. {message}\nФайл: {output_path}",
                )
            except Exception as exc:
                LOGGER.exception("Image inference failed")
                messagebox.showerror("Ошибка", str(exc))

        ttk.Button(frame, text="Запустить инференс", command=run_infer).grid(
            row=7, column=0, columnspan=2, pady=14
        )
        return frame

    def _build_infer_video_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Инференс по видео/камере", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(6, 10), padx=8
        )
        weights = self._row(frame, "Весы модели", "weights/best.pt", 1)
        source = self._row(frame, "Путь к видео или RTSP", "data/test.mp4", 2)
        output = self._row(frame, "Путь для сохранения (пусто = не сохранять)", "", 3)
        conf = self._row(frame, "Confidence", "0.25", 4)
        display = self._row(frame, "Показывать окно? (y/n)", "y", 5)

        def run_infer() -> None:
            try:
                infer_video(
                    weights=weights.get(),
                    source=source.get(),
                    conf=float(conf.get()),
                    display=display.get().lower().startswith("y"),
                    output=output.get() or None,
                )
                messagebox.showinfo("Готово", "Инференс завершен.")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

        ttk.Button(frame, text="Запустить инференс", command=run_infer).grid(
            row=6, column=0, columnspan=2, pady=14
        )
        return frame


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
