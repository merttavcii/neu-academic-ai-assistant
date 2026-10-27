"""DuckDuckGo metin araması — kısa snippet'leri Groq'a bağlam olarak verir."""

from __future__ import annotations

import os


def get_web_snippets(query: str, *, max_results: int = 5, max_chars: int = 4000) -> str:
    """
    Soru metni için DDG metin sonuçlarını blok halinde döndürür.
    Kapatmak için: RAG_WEB_SEARCH=0
    """
    flag = os.environ.get("RAG_WEB_SEARCH", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return ""
    q = (query or "").strip()
    if len(q) < 2:
        return ""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return ""

    lines: list[str] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=max_results):
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                href = (r.get("href") or "").strip()
                if not body and not title:
                    continue
                lines.append(f"Başlık: {title}\nÖzet: {body}\nURL: {href}")
    except Exception:
        return ""

    out = "\n\n---\n\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n\n[... web özeti kısaltıldı ...]"
    return out.strip()
