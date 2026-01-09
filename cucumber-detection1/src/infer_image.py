from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output", default=None, help="Optional output path.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--display", action="store_true", help="Show image preview window.")
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay for preview window.")
    return parser.parse_args()


def infer_image(
    *,
    weights: str,
    image_path: str,
    output: str | None,
    conf: float,
    display: bool = False,
    delay_ms: int = 0,
) -> tuple[Path, bool]:
    model = YOLO(weights)
    results = model.predict(source=image_path, conf=conf, save=False, verbose=False)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    detected = False
    for result in results:
        for box in result.boxes:
            detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = "Огурец: да" if detected else "Огурец: нет"
    cv2.putText(
        image,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    output_path = Path(output) if output else Path(image_path).with_suffix(".detected.jpg")
    cv2.imwrite(str(output_path), image)
    print(f"Saved to {output_path}")
    if display:
        window_name = "Инференс: огурец"
        cv2.imshow(window_name, image)
        wait = 0 if delay_ms <= 0 else delay_ms
        cv2.waitKey(wait)
        cv2.destroyAllWindows()
    return output_path, detected


def main() -> None:
    args = parse_args()
    infer_image(
        weights=args.weights,
        image_path=args.image,
        output=args.output,
        conf=args.conf,
        display=args.display,
        delay_ms=args.delay_ms,
    )


if __name__ == "__main__":
    main()
