from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on video or camera stream.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument("--source", required=True, help="Video path or camera URL.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--display", action="store_true", help="Show window with detections.")
    parser.add_argument("--output", default=None, help="Optional output video path.")
    return parser.parse_args()


def infer_video(
    *,
    weights: str,
    source: str,
    conf: float,
    display: bool,
    output: str | None,
) -> None:
    model = YOLO(weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    writer = None
    if output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=conf, save=False, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if writer is not None:
                writer.write(frame)

            if display:
                cv2.imshow("Cucumber Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    infer_video(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        display=args.display,
        output=args.output,
    )


if __name__ == "__main__":
    main()
