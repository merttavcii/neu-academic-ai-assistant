from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from statistics import median
from typing import Any


CONF_THRESHOLD = 0.50
POSITION_MARGIN = 0.08
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.50
SEQUENCE_SMALL_SWAP_MAX_SHIFT = 1
SEQUENCE_SMALL_SWAP_MAX_MISPLACED = 4

UPPER_EXPECTED_ORDER = list(range(1, 17))
LOWER_EXPECTED_ORDER = list(range(32, 16, -1))


def assign_jaw(tooth_number: int) -> str:
    if 1 <= tooth_number <= 16:
        return "upper"
    if 17 <= tooth_number <= 32:
        return "lower"
    return "unknown"


def filter_duplicate_teeth(predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unknown_predictions: list[dict[str, Any]] = []

    for prediction in predictions:
        tooth_number = prediction.get("tooth_number")
        if isinstance(tooth_number, int):
            grouped[tooth_number].append(prediction)
        else:
            unknown_predictions.append(prediction)

    filtered: list[dict[str, Any]] = []
    removed_duplicates: list[dict[str, Any]] = []

    for tooth_number in sorted(grouped):
        candidates = sorted(grouped[tooth_number], key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        kept = candidates[0]
        filtered.append(kept)

        for removed in candidates[1:]:
            removed_duplicates.append(
                {
                    "tooth_number": tooth_number,
                    "kept_confidence": round(float(kept.get("confidence", 0.0)), 4),
                    "removed_confidence": round(float(removed.get("confidence", 0.0)), 4),
                    "reason": "duplicate_tooth_number_lower_confidence_removed",
                }
            )

    filtered.extend(unknown_predictions)
    return filtered, removed_duplicates


def validate_anatomical_position(
    predictions: list[dict[str, Any]],
    conf_threshold: float = CONF_THRESHOLD,
    position_margin: float = POSITION_MARGIN,
) -> list[dict[str, Any]]:
    upper_y = [_center_norm_y(item) for item in predictions if item.get("jaw") == "upper"]
    lower_y = [_center_norm_y(item) for item in predictions if item.get("jaw") == "lower"]
    upper_y = [value for value in upper_y if value is not None]
    lower_y = [value for value in lower_y if value is not None]

    if upper_y and lower_y:
        upper_median_y = median(upper_y)
        lower_median_y = median(lower_y)
        jaw_boundary = (upper_median_y + lower_median_y) / 2
    else:
        jaw_boundary = 0.5

    for prediction in predictions:
        tooth_number = int(prediction.get("tooth_number", -1))
        jaw = prediction.get("jaw")
        center_y = _center_norm_y(prediction)
        reasons: list[str] = []

        if float(prediction.get("confidence", 0.0)) < conf_threshold:
            reasons.append("low_confidence")

        if jaw == "upper" and center_y is not None and center_y > jaw_boundary + position_margin:
            reasons.append("upper_too_low")

        if jaw == "lower" and center_y is not None and center_y < jaw_boundary - position_margin:
            reasons.append("lower_too_high")

        if 1 <= tooth_number <= 16 and jaw != "upper":
            reasons.append("number_jaw_mismatch")
        elif 17 <= tooth_number <= 32 and jaw != "lower":
            reasons.append("number_jaw_mismatch")

        prediction["anatomy_status"] = "suspicious" if reasons else "valid"
        prediction["anatomy_reasons"] = reasons or ["ok"]

    return predictions


def validate_sequence(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "upper": _validate_jaw_sequence(predictions, "upper", UPPER_EXPECTED_ORDER),
        "lower": _validate_jaw_sequence(predictions, "lower", LOWER_EXPECTED_ORDER),
    }


def apply_sequence_results_to_predictions(
    predictions: list[dict[str, Any]],
    sequence_summary: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    for prediction in predictions:
        jaw = prediction.get("jaw")
        tooth_number = int(prediction.get("tooth_number", -1))

        jaw_summary = sequence_summary.get(jaw) if isinstance(jaw, str) else None

        if not jaw_summary:
            prediction["sequence_level"] = "serious_sequence_conflict"
            prediction["sequence_detail"] = "unknown_jaw"

            prediction["jaw_sequence_level"] = "serious_sequence_conflict"
            prediction["jaw_sequence_detail"] = "unknown_jaw"
            prediction["jaw_detected_order"] = []
            prediction["jaw_expected_detected_order"] = []
            prediction["missing_possible_in_jaw"] = []
            prediction["misplaced_teeth_in_jaw"] = []
            continue

        misplaced_teeth = jaw_summary.get("misplaced_teeth", [])

        # Çene seviyesi bilgi: tüm dişlerde kalabilir
        prediction["jaw_sequence_level"] = jaw_summary.get("level", "valid")
        prediction["jaw_sequence_detail"] = jaw_summary.get("detail", "order_ok")
        prediction["jaw_detected_order"] = jaw_summary.get("detected_order", [])
        prediction["jaw_expected_detected_order"] = jaw_summary.get("expected_detected_order", [])
        prediction["missing_possible_in_jaw"] = jaw_summary.get("missing_possible", [])
        prediction["misplaced_teeth_in_jaw"] = misplaced_teeth

        # Diş seviyesi uyarı: sadece gerçekten sırası bozulan dişlere verilsin
        if tooth_number in misplaced_teeth:
            prediction["sequence_level"] = jaw_summary.get("level", "valid")
            prediction["sequence_detail"] = jaw_summary.get("detail", "order_ok")
        else:
            prediction["sequence_level"] = "valid"
            prediction["sequence_detail"] = "order_ok"

    return predictions


def process_tooth_predictions(
    predictions: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    raw_count = len(predictions)
    working = [_standardize_prediction(item, image_width, image_height) for item in deepcopy(predictions)]
    filtered, removed_duplicates = filter_duplicate_teeth(working)

    for prediction in filtered:
        prediction["jaw"] = assign_jaw(int(prediction.get("tooth_number", -1)))

    validate_anatomical_position(filtered)
    sequence_summary = validate_sequence(filtered)
    apply_sequence_results_to_predictions(filtered, sequence_summary)

    valid_count = sum(1 for item in filtered if item.get("anatomy_status") == "valid")
    suspicious_count = sum(1 for item in filtered if item.get("anatomy_status") == "suspicious")
    upper_count = sum(1 for item in filtered if item.get("jaw") == "upper")
    lower_count = sum(1 for item in filtered if item.get("jaw") == "lower")

    anatomy_summary = {
        "total_teeth_detected_raw": raw_count,
        "total_teeth_after_duplicate_filter": len(filtered),
        "removed_duplicate_count": len(removed_duplicates),
        "valid_teeth_count": valid_count,
        "suspicious_teeth_count": suspicious_count,
        "upper_count": upper_count,
        "lower_count": lower_count,
        "sequence": sequence_summary,
    }

    return {
        "teeth": filtered,
        "removed_duplicates": removed_duplicates,
        "anatomy_summary": anatomy_summary,
    }


def _standardize_prediction(
    prediction: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    bbox = prediction.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    prediction["tooth_number"] = int(prediction.get("tooth_number", int(prediction.get("class_id", -1)) + 1))
    prediction["class_id"] = int(prediction.get("class_id", prediction["tooth_number"] - 1))
    prediction["confidence"] = float(prediction.get("confidence", 0.0))
    prediction["bbox"] = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
    prediction["bbox_xyxy"] = {
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
    }
    prediction["center"] = {"x": round(center_x, 2), "y": round(center_y, 2)}
    prediction["center_norm"] = {
        "x": _safe_norm(center_x, image_width),
        "y": _safe_norm(center_y, image_height),
    }
    prediction.setdefault("polygon", [])
    prediction.setdefault("jaw", None)
    prediction.setdefault("anatomy_status", None)
    prediction.setdefault("anatomy_reasons", [])
    prediction.setdefault("sequence_level", None)
    prediction.setdefault("sequence_detail", None)
    return prediction


def _validate_jaw_sequence(
    predictions: list[dict[str, Any]],
    jaw: str,
    expected_order: list[int],
) -> dict[str, Any]:
    jaw_predictions = [item for item in predictions if item.get("jaw") == jaw]

    if not jaw_predictions:
        return {
            "level": "serious_sequence_conflict",
            "detected_order": [],
            "expected_detected_order": [],
            "missing_possible": expected_order[:],
            "misplaced_teeth": [],
            "detail": "no_teeth_detected",
        }

    sorted_predictions = sorted(
        jaw_predictions,
        key=lambda item: float(item.get("center_norm", {}).get("x", 0.0))
    )

    detected_order = [int(item["tooth_number"]) for item in sorted_predictions]
    detected_set = set(detected_order)

    expected_detected_order = [
        tooth for tooth in expected_order
        if tooth in detected_set
    ]

    missing_possible = [
        tooth for tooth in expected_order
        if tooth not in detected_set
    ]

    if detected_order == expected_detected_order:
        return {
            "level": "valid",
            "detected_order": detected_order,
            "expected_detected_order": expected_detected_order,
            "missing_possible": missing_possible,
            "misplaced_teeth": [],
            "detail": "order_ok",
        }

    expected_index = {
        tooth: index
        for index, tooth in enumerate(expected_detected_order)
    }

    misplaced: list[int] = []
    max_shift = 0

    for actual_index, tooth in enumerate(detected_order):
        if tooth not in expected_index:
            misplaced.append(tooth)
            max_shift = max(max_shift, 99)
            continue

        shift = abs(actual_index - expected_index[tooth])

        if shift > 0:
            misplaced.append(tooth)
            max_shift = max(max_shift, shift)

    if (
        max_shift <= SEQUENCE_SMALL_SWAP_MAX_SHIFT
        and len(misplaced) <= SEQUENCE_SMALL_SWAP_MAX_MISPLACED
    ):
        level = "mild_sequence_warning"
        detail = f"small_neighbor_swap: {misplaced}"
    else:
        level = "serious_sequence_conflict"
        detail = f"large_order_conflict: {misplaced}"

    return {
        "level": level,
        "detected_order": detected_order,
        "expected_detected_order": expected_detected_order,
        "missing_possible": missing_possible,
        "misplaced_teeth": misplaced,
        "detail": detail,
    }


def _center_norm_y(prediction: dict[str, Any]) -> float | None:
    center_norm = prediction.get("center_norm") or {}
    value = center_norm.get("y")
    if value is None:
        return None
    return float(value)


def _safe_norm(value: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(value) / float(denominator), 6)
