from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


TOOTH_COLOR = (80, 200, 120)
TOOTH_MILD_SEQUENCE_COLOR = (42, 210, 255)
TOOTH_SUSPICIOUS_COLOR = (28, 140, 255)
TOOTH_SERIOUS_SEQUENCE_COLOR = (68, 68, 255)
FINDING_COLORS = {
    "Crown": (64, 180, 255),
    "Filling": (46, 204, 113),
    "Implant": (255, 120, 88),
    "Root Canal Treatment": (170, 112, 255),
    "Root Piece": (68, 92, 255),
    "impacted tooth": (255, 92, 190),
}


def _draw_mask_overlay(image: np.ndarray, mask: Any, color: tuple[int, int, int], alpha: float) -> None:
    if mask is None:
        return

    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr.squeeze()
    if mask_arr.ndim != 2 or mask_arr.size == 0:
        return

    height, width = image.shape[:2]
    if mask_arr.shape[:2] != (height, width):
        mask_arr = cv2.resize(mask_arr.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)

    binary = mask_arr > 0.5
    if not binary.any():
        return

    overlay = image.copy()
    overlay[binary] = color
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def _label(image: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    y = max(22, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - text_h - 10), (x + text_w + 12, y + 6), (12, 21, 34), -1)
    cv2.rectangle(image, (x, y - text_h - 10), (x + text_w + 12, y + 6), color, 1)
    cv2.putText(image, text, (x + 6, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _bbox_anchor(item: dict[str, Any]) -> tuple[int, int]:
    bbox = item.get("bbox") or [8, 24, 8, 24]
    return int(bbox[0]), int(bbox[1])


def _tooth_color(tooth: dict[str, Any]) -> tuple[int, int, int]:
    if tooth.get("sequence_level") == "serious_sequence_conflict":
        return TOOTH_SERIOUS_SEQUENCE_COLOR
    if tooth.get("sequence_level") == "mild_sequence_warning":
        return TOOTH_MILD_SEQUENCE_COLOR
    if tooth.get("anatomy_status") == "suspicious":
        return TOOTH_SUSPICIOUS_COLOR
    return TOOTH_COLOR


def _tooth_label(tooth: dict[str, Any]) -> str:
    tooth_number = tooth.get("tooth_number", "?")
    if tooth.get("sequence_level") == "serious_sequence_conflict":
        return f"{tooth_number} !"
    if tooth.get("sequence_level") == "mild_sequence_warning":
        return f"{tooth_number} swap"
    if tooth.get("anatomy_status") == "suspicious":
        return f"{tooth_number} !"
    return str(tooth_number)


def create_annotated_image(
    image_path: Path,
    teeth: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read uploaded image for visualization.")

    annotated = image.copy()

    for tooth in teeth:
        _draw_mask_overlay(annotated, tooth.get("mask"), _tooth_color(tooth), 0.18)

    for finding in findings:
        name = finding.get("finding_name", "Finding")
        _draw_mask_overlay(annotated, finding.get("mask"), FINDING_COLORS.get(name, (80, 220, 220)), 0.42)

    for tooth in teeth:
        x, y = _bbox_anchor(tooth)
        _label(annotated, _tooth_label(tooth), (x, y), _tooth_color(tooth))

    for finding in findings:
        x, y = _bbox_anchor(finding)
        name = finding.get("finding_name", "Finding")
        confidence = int(round(float(finding.get("confidence", 0.0)) * 100))
        _label(annotated, f"{name} {confidence}%", (x, y + 26), FINDING_COLORS.get(name, (80, 220, 220)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
