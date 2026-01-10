from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


def _check_path(label: str, path: Path) -> bool:
    if path.exists():
        print(f"[OK] {label}: {path}")
        return True
    print(f"[MISSING] {label}: {path}")
    return False


def _check_module(module: str, install_hint: str | None = None) -> bool:
    if find_spec(module) is not None:
        print(f"[OK] module: {module}")
        return True
    hint = f" ({install_hint})" if install_hint else ""
    print(f"[MISSING] module: {module}{hint}")
    return False


def main() -> None:
    root = Path(__file__).resolve().parent
    print("=== Быстрая проверка проекта ===")

    checks = [
        _check_path("Папка src", root / "src"),
        _check_path("Файл configs/cucumber_prior.yaml", root / "configs" / "cucumber_prior.yaml"),
        _check_path("Файл configs/dataset.yaml", root / "configs" / "dataset.yaml"),
        _check_path("Папка weights", root / "weights"),
        _check_path("Папка auto_train", root / "auto_train"),
    ]

    module_checks = [
        _check_module("ultralytics", "pip install ultralytics"),
        _check_module("cv2", "pip install opencv-python"),
        _check_module("torch", "pip install torch"),
        _check_module("yaml", "pip install pyyaml"),
        _check_module("numpy", "pip install numpy"),
    ]

    ok = all(checks) and all(module_checks)
    print("\n=== Итог ===")
    if ok:
        print("Готово к запуску.")
    else:
        print("Есть проблемы — см. выше.")


if __name__ == "__main__":
    main()
