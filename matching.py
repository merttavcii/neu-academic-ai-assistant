from __future__ import annotations

from typing import Any

import cv2
import numpy as np


DEFAULT_OVERLAP_THRESHOLD = 0.05


EXPLANATIONS = {
    "Crown": "A crown-like restoration was detected on this tooth.",
    "Filling": "A filling-like restoration was detected on this tooth.",
    "Implant": "An implant-like structure was detected.",
    "Root Canal Treatment": "Signs similar to root canal treatment were detected.",
    "Root Piece": "A root fragment-like structure was detected.",
    "impacted tooth": "An impacted tooth-like structure was detected.",
}


def explanation_for(finding_name: str, tooth_number: int | None) -> str:
    base = EXPLANATIONS.get(finding_name, "A dental finding was detected.")
    if tooth_number is None:
        return base
    return base.replace("this tooth", f"tooth {tooth_number}")


def _as_binary_mask(mask: Any, target_shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None

    arr = np.asarray(mask)
    if arr.size == 0:
        return None

    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.ndim != 2:
        return None

    target_h, target_w = target_shape
    if arr.shape[:2] != (target_h, target_w):
        arr = cv2.resize(arr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return (arr > 0.5).astype(np.uint8)


def intersection_ratio(finding_mask: Any, tooth_mask: Any, image_shape: tuple[int, int]) -> float:
    finding = _as_binary_mask(finding_mask, image_shape)
    tooth = _as_binary_mask(tooth_mask, image_shape)
    if finding is None or tooth is None:
        return 0.0

    finding_area = int(finding.sum())
    if finding_area == 0:
        return 0.0

    overlap = int(np.logical_and(finding, tooth).sum())
    return float(overlap / finding_area)


def match_findings_to_teeth(
    teeth: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    image_shape: tuple[int, int],
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []

    for finding in findings:
        best_tooth = None
        best_score = 0.0

        for tooth in teeth:
            score = intersection_ratio(finding.get("mask"), tooth.get("mask"), image_shape)
            if score > best_score:
                best_score = score
                best_tooth = tooth

        tooth_number = None
        matched_tooth_anatomy_status = None
        matched_tooth_sequence_level = None
        matched_tooth_anatomy_reasons: list[str] = []
        if best_tooth is not None and best_score >= threshold:
            tooth_number = best_tooth.get("tooth_number")
            matched_tooth_anatomy_status = best_tooth.get("anatomy_status")
            matched_tooth_sequence_level = best_tooth.get("sequence_level")
            matched_tooth_anatomy_reasons = list(best_tooth.get("anatomy_reasons") or [])

        finding_name = finding.get("finding_name", "Unknown")
        matched.append(
            {
                "tooth_number": tooth_number,
                "matched_tooth_number": tooth_number,
                "finding": finding_name,
                "finding_class": finding_name,
                "confidence": finding.get("confidence", 0.0),
                "overlap_score": round(best_score, 4),
                "matched_tooth_anatomy_status": matched_tooth_anatomy_status,
                "matched_tooth_sequence_level": matched_tooth_sequence_level,
                "matched_tooth_anatomy_reasons": matched_tooth_anatomy_reasons,
                "explanation": explanation_for(finding_name, tooth_number),
            }
        )

    return matched
