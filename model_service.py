from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import cv2
import numpy as np

from anatomy_engine import YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, process_tooth_predictions

TOOTH_IMGSZ = 1024
TOOTH_CONF = YOLO_CONF_THRESHOLD
TOOTH_IOU = YOLO_IOU_THRESHOLD
FINDING_IMGSZ = 768
FINDING_CONF = 0.25
FINDING_IOU = YOLO_IOU_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TOOTH_MODEL_PATH = MODEL_DIR / "best_YOLOv11m_tooth_detection.pt"
FINDING_MODEL_PATH = MODEL_DIR / "best_YOLOv11.pt"

logger = logging.getLogger(__name__)

FINDING_CLASSES = {
    0: "Crown",
    1: "Filling",
    2: "Implant",
    3: "Root Canal Treatment",
    4: "Root Piece",
    5: "impacted tooth",
}


class ModelService:
    def __init__(self) -> None:
        self.tooth_model = None
        self.finding_model = None
        self.load_error: str | None = None

    def load(self) -> None:
        missing = [str(path) for path in (TOOTH_MODEL_PATH, FINDING_MODEL_PATH) if not path.exists()]
        if missing:
            self.load_error = "Missing model file(s): " + ", ".join(missing)
            return

        try:
            from ultralytics import YOLO

            self.tooth_model = YOLO(str(TOOTH_MODEL_PATH))
            self.finding_model = YOLO(str(FINDING_MODEL_PATH))
            self.load_error = None
            logger.info("Models loaded successfully: tooth_model=%s finding_model=%s", TOOTH_MODEL_PATH, FINDING_MODEL_PATH)
        except Exception as exc:
            self.load_error = f"Model loading failed: {exc}"
            logger.exception("Model loading failed")

    @property
    def ready(self) -> bool:
        return self.tooth_model is not None and self.finding_model is not None and self.load_error is None

    def analyze(self, image_path: Path) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError(self.load_error or "Models are not loaded.")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Uploaded file could not be read as an image.")

        height, width = image.shape[:2]

        try:
            tooth_result = self.tooth_model.predict(
                str(image_path),
                imgsz=TOOTH_IMGSZ,
                conf=TOOTH_CONF,
                iou=TOOTH_IOU,
                verbose=False,
            )[0]
            finding_result = self.finding_model.predict(
                str(image_path),
                imgsz=FINDING_IMGSZ,
                conf=FINDING_CONF,
                iou=FINDING_IOU,
                verbose=False,
            )[0]
        except Exception as exc:
            raise RuntimeError(f"Inference failed: {exc}") from exc

        raw_teeth = _extract_detections(
            tooth_result,
            image_shape=(height, width),
            label_mode="tooth",
        )
        findings = _extract_detections(
            finding_result,
            image_shape=(height, width),
            label_mode="finding",
        )

        anatomy_result = process_tooth_predictions(raw_teeth, image_width=width, image_height=height)
        teeth = anatomy_result["teeth"]
        summary = anatomy_result["anatomy_summary"]
        logger.info(
            "Anatomy pipeline: raw_teeth=%s filtered_teeth=%s duplicates_removed=%s suspicious=%s upper=%s lower=%s sequence_upper=%s sequence_lower=%s findings=%s",
            summary["total_teeth_detected_raw"],
            summary["total_teeth_after_duplicate_filter"],
            summary["removed_duplicate_count"],
            summary["suspicious_teeth_count"],
            summary["upper_count"],
            summary["lower_count"],
            summary["sequence"]["upper"]["level"],
            summary["sequence"]["lower"]["level"],
            len(findings),
        )

        return {
            "teeth": teeth,
            "findings": findings,
            "image_shape": (height, width),
            "removed_duplicates": anatomy_result["removed_duplicates"],
            "anatomy_summary": summary,
        }


def _extract_detections(result: Any, image_shape: tuple[int, int], label_mode: str) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    masks = getattr(result, "masks", None)
    mask_data = None
    if masks is not None and getattr(masks, "data", None) is not None:
        mask_data = masks.data.detach().cpu().numpy()

    detections: list[dict[str, Any]] = []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)

    for index, class_id in enumerate(classes):
        bbox = [round(float(v), 2) for v in xyxy[index].tolist()]
        mask = None
        polygon: list[list[float]] = []

        if mask_data is not None and index < len(mask_data):
            mask = _resize_mask(mask_data[index], image_shape)
            polygon = _mask_to_polygon(mask)

        confidence = round(float(confs[index]), 4)
        if label_mode == "tooth":
            detections.append(
                {
                    "tooth_number": int(class_id) + 1,
                    "confidence": confidence,
                    "bbox": bbox,
                    "mask": mask,
                    "polygon": polygon,
                    "class_id": int(class_id),
                }
            )
        else:
            detections.append(
                {
                    "finding_name": FINDING_CLASSES.get(int(class_id), f"Class {class_id}"),
                    "confidence": confidence,
                    "bbox": bbox,
                    "mask": mask,
                    "polygon": polygon,
                    "class_id": int(class_id),
                }
            )

    return detections


def _resize_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.uint8)


def _mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [[float(point[0][0]), float(point[0][1])] for point in approx]


def strip_internal_masks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_items = []
    for item in items:
        public_item = dict(item)
        public_item.pop("mask", None)
        public_items.append(public_item)
    return public_items


model_service = ModelService()
