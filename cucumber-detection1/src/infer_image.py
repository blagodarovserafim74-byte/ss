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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(source=args.image, conf=args.conf, save=False, verbose=False)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_path = Path(args.output) if args.output else Path(args.image).with_suffix(".detected.jpg")
    cv2.imwrite(str(output_path), image)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
