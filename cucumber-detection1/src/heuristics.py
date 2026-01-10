from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CucumberPrior:
    lower_hsv: Tuple[int, int, int]
    upper_hsv: Tuple[int, int, int]
    min_area: int
    max_area: int
    min_aspect_ratio: float
    max_aspect_ratio: float
    min_extent: float
    max_extent: float
    min_solidity: float


@dataclass(frozen=True)
class Detection:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_yolo(self, width: int, height: int) -> Tuple[float, float, float, float]:
        x_center = (self.x_min + self.x_max) / 2.0 / width
        y_center = (self.y_min + self.y_max) / 2.0 / height
        box_width = (self.x_max - self.x_min) / width
        box_height = (self.y_max - self.y_min) / height
        return x_center, y_center, box_width, box_height


def load_prior(raw: dict) -> CucumberPrior:
    return CucumberPrior(
        lower_hsv=tuple(raw["color_hsv"]["lower"]),
        upper_hsv=tuple(raw["color_hsv"]["upper"]),
        min_area=int(raw["shape"]["min_area"]),
        max_area=int(raw["shape"]["max_area"]),
        min_aspect_ratio=float(raw["shape"]["min_aspect_ratio"]),
        max_aspect_ratio=float(raw["shape"]["max_aspect_ratio"]),
        min_extent=float(raw["shape"]["min_extent"]),
        max_extent=float(raw["shape"]["max_extent"]),
        min_solidity=float(raw["shape"]["min_solidity"]),
    )


def _filter_contour(contour: np.ndarray, prior: CucumberPrior) -> bool:
    area = cv2.contourArea(contour)
    if area < prior.min_area or area > prior.max_area:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    if h == 0 or w == 0:
        return False

    aspect_ratio = max(w, h) / min(w, h)
    if not (prior.min_aspect_ratio <= aspect_ratio <= prior.max_aspect_ratio):
        return False

    rect_area = float(w * h)
    extent = area / rect_area
    if not (prior.min_extent <= extent <= prior.max_extent):
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False

    solidity = area / hull_area
    return solidity >= prior.min_solidity


def find_cucumber_candidates(image_bgr: np.ndarray, prior: CucumberPrior) -> List[Detection]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(prior.lower_hsv), np.array(prior.upper_hsv))
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Detection] = []
    for contour in contours:
        if _filter_contour(contour, prior):
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection(x, y, x + w, y + h))
    return detections


def draw_detections(image_bgr: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    output = image_bgr.copy()
    for det in detections:
        cv2.rectangle(output, (det.x_min, det.y_min), (det.x_max, det.y_max), (0, 255, 0), 2)
    return output
