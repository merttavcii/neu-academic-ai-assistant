"""Diş hekimliği sorularını ana asistandan dental servise yönlendirme kuralları."""

from __future__ import annotations

import re

_DENTAL_HINTS = (
    "diş", "dis ", " dis", "dişler", "disler", "dişin", "disin",
    "dişe", "dise", "dişi", "disi", "dişte", "diste", "dişten", "disten",
    "çürük", "curuk", "çürüğ", "curug", "dolgu", "kanal tedavisi",
    "endodonti", "periodont", "diş eti", "dis eti", "gingiva",
    "implant", "kron", "kaplama", "protez", "ortodonti", "braket",
    "panoramik", "periapikal", "bitewing", "röntgen", "rontgen",
    "x-ray", "xray", "x ray", "opg", "cbct", "tomografi",
    "FDI", "FDI numarası", "diş numarası", "dis numarasi",
    "gömülü diş", "gomulu dis", "impacted tooth", "impacted",
    "kök kanalı", "kok kanali", "root canal", "pulpa", "pulpectomy",
    "karies", "caries", "restorasyon", "amalgam", "kompozit",
    "molar", "premolar", "kanin", "köpek dişi", "azı dişi",
    "mandibula", "maksilla", "maxilla", "alveol", "cementum",
    "enamel", "mine", "dentin", "odont", "oral", "ağız", "agiz",
    "dental", "dentist", "dentistry", "tooth", "teeth",
    "filling", "crown", "extraction", "extraction",
    "periodontal", "gingivitis", "abscess", "apical",
    "radiograph", "radiography", "orthodontic", "prosthodontic",
    "endodontic", "pedodont", "oral surgery", "oral surgeon",
    "scaler", "tartar", "plaque", "calculus", "fluoride",
)

_DENTAL_IMAGE_HINTS = (
    "röntgen", "rontgen", "x-ray", "xray", "panoramik", "opg",
    "periapikal", "bitewing", "cbct", "tomografi", "radiograph",
)

_NON_DENTAL_OVERRIDE = (
    "kemik kırığı", "kemik kirigi", "fracture", "radius", "ulna",
    "femur", "tibia", "el röntgeni", "ayak röntgeni", "göğüs röntgeni",
    "chest x-ray", "chest xray",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_dental_image_request(*, question: str = "", file_kind: str = "") -> bool:
    if file_kind == "dental":
        return True
    t = _normalize(question)
    return any(h in t for h in _DENTAL_IMAGE_HINTS)


def should_route_to_dental(
    message: str,
    *,
    dental_context: dict | None = None,
    dental_mode: bool = False,
) -> bool:
    """
    True → istek dental FastAPI servisine gider.
    Aktif dental analiz bağlamı veya açık dental mod varsa devam eder.
    """
    t = _normalize(message)

    if dental_context and isinstance(dental_context, dict):
        has_data = bool(
            dental_context.get("matches")
            or dental_context.get("matched_results")
            or dental_context.get("findings")
            or dental_context.get("teeth")
            or dental_context.get("tooth_predictions")
        )
        if has_data and t and not any(h in t for h in _NON_DENTAL_OVERRIDE):
            return True

    if dental_mode and t:
        return True

    if not t:
        return False

    if any(h in t for h in _NON_DENTAL_OVERRIDE):
        return False

    return any(h in t for h in _DENTAL_HINTS)
