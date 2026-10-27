"""
dental_client.py
-----------------
Flask uygulamasının (app.py), ayrı bir process olarak port 8010'da çalışan
"DentAI Vision" FastAPI servisiyle (main.py) konuşmasını sağlayan ince
HTTP istemci katmanı.

DentAI Vision servisi şu endpoint'leri sunar:
  - GET  /health   -> {"ok": bool, "models_ready": bool, "model_error": ..., "disclaimer": ...}
  - POST /analyze  -> multipart "file" alanı; JSON analiz sonucu döner
  - POST /chat     -> JSON {"question", "analysis_context", "history"}; {"answer": ...} döner

Bu dosya app.py tarafından şu fonksiyonlarla kullanılır:
  - get_base_url()
  - health_check()
  - analyze_xray(raw, filename, mime)
  - format_analysis_summary(payload, question=...)
  - build_analysis_context(payload)
  - chat(question, analysis_context=None, history=None)
"""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8010"
DEFAULT_TIMEOUT_SECONDS = 15
ANALYZE_TIMEOUT_SECONDS = 120
CHAT_TIMEOUT_SECONDS = 60


def get_base_url() -> str:
    """DentAI Vision servisinin taban URL'sini döner (env ile override edilebilir)."""
    return os.getenv("DENTAL_SERVICE_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def health_check() -> dict[str, Any]:
    """DentAI Vision servisinin /health uç noktasını kontrol eder."""
    url = f"{get_base_url()}/health"
    resp = requests.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def analyze_xray(raw: bytes, filename: str, mime: str) -> dict[str, Any]:
    """
    Diş röntgeni görselini DentAI Vision servisinin /analyze uç noktasına
    gönderir ve ham JSON analiz sonucunu (payload) döner.
    """
    url = f"{get_base_url()}/analyze"
    files = {"file": (filename or "dental_xray.png", raw, mime or "image/png")}

    resp = requests.post(url, files=files, timeout=ANALYZE_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(payload.get("error") or "Dental analiz servisi bir hata döndürdü.")

    return payload


def build_analysis_context(payload: dict[str, Any]) -> dict[str, Any]:
    """
    /analyze cevabından, sonraki /chat çağrılarında ve sohbet geçmişinde
    kullanılacak sade bir bağlam (context) sözlüğü oluşturur.
    """
    payload = payload or {}
    matched_results = payload.get("matched_results") or payload.get("matches") or []

    return {
        "matched_results": matched_results,
        "matches": matched_results,
        "findings": payload.get("findings") or [],
        "teeth": payload.get("teeth") or payload.get("tooth_predictions") or [],
        "ai_explanation": payload.get("ai_explanation") or "",
        "warnings": payload.get("warnings") or [],
        "anatomy_summary": payload.get("anatomy_summary") or {},
        "disclaimer": payload.get("disclaimer") or "",
        "image_id": payload.get("image_id"),
        "original_image": payload.get("original_image"),
        "annotated_image": payload.get("annotated_image") or payload.get("annotated_image_url"),
    }


def _group_matches_by_finding(matched_results: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in matched_results or []:
        if not isinstance(item, dict):
            continue
        finding = str(item.get("finding") or item.get("finding_class") or "").strip()
        tooth = item.get("tooth_number", item.get("matched_tooth_number"))
        if not finding or tooth is None:
            continue
        grouped.setdefault(finding, [])
        tooth_str = str(tooth).strip()
        if tooth_str not in grouped[finding]:
            grouped[finding].append(tooth_str)
    return grouped


def format_analysis_summary(payload: dict[str, Any], question: str = "") -> str:
    """
    /analyze sonucunu, sohbet arayüzünde gösterilecek okunabilir Türkçe bir
    özet metnine dönüştürür.
    """
    payload = payload or {}
    teeth = payload.get("teeth") or payload.get("tooth_predictions") or []
    findings = payload.get("findings") or []
    matched_results = payload.get("matched_results") or payload.get("matches") or []
    warnings = payload.get("warnings") or []
    disclaimer = payload.get("disclaimer") or ""

    lines: list[str] = ["**Diş Röntgeni Analiz Sonucu**", ""]
    lines.append(f"- Tespit edilen diş sayısı: {len(teeth)}")
    lines.append(f"- Tespit edilen bulgu sayısı: {len(findings)}")

    grouped = _group_matches_by_finding(matched_results)
    if grouped:
        lines.append("")
        lines.append("**Bulgular (diş numarasına göre):**")
        for finding, teeth_list in grouped.items():
            teeth_sorted = sorted(
                teeth_list,
                key=lambda v: (0, int(v)) if v.isdigit() else (1, v),
            )
            lines.append(f"- {finding}: diş {', '.join(teeth_sorted)}")
    else:
        lines.append("")
        lines.append("Görselde belirgin bir bulgu-diş eşleşmesi tespit edilmedi.")

    if warnings:
        lines.append("")
        lines.append("**Uyarılar:**")
        for w in warnings:
            lines.append(f"- {w}")

    if disclaimer:
        lines.append("")
        lines.append(f"_{disclaimer}_")

    return "\n".join(lines)


def chat(
    question: str,
    analysis_context: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Kullanıcının analiz sonrası dental sorusunu DentAI Vision servisinin
    /chat uç noktasına iletir ve modelin cevabını döner.
    """
    url = f"{get_base_url()}/chat"
    body = {
        "question": (question or "").strip(),
        "analysis_context": analysis_context or {},
        "history": [
            {"role": h.get("role"), "content": h.get("content")}
            for h in (history or [])
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content")
        ],
    }

    resp = requests.post(url, json=body, timeout=CHAT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    return (data or {}).get("answer") or ""