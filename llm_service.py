from __future__ import annotations

import logging
import os
import re
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import requests
except ImportError:
    requests = None


logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma2:2b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 300
DEFAULT_CHAT_TIMEOUT_SECONDS = 120

SAFETY_SENTENCE = (
    "AI-generated results can have false positives or false negatives and are not a diagnosis; please review them with a licensed dental professional."
)

CHAT_FALLBACK_RESPONSE = (
    "The chatbot is unavailable right now. AI results can have false positives or false negatives and are not a diagnosis."
)

NO_FINDINGS_EXPLANATION = (
    "No specific dental findings were detected by the AI analysis in this image. "
    "However, this AI-generated result may include false positives or false negatives, "
    "is not a diagnosis, and should still be reviewed by a licensed dental professional."
)

DISABLED_EXPLANATION = (
    "AI explanation is currently disabled. The detection results are AI-generated, "
    "may include false positives or false negatives, are not a diagnosis, and should be reviewed "
    "by a licensed dental professional."
)

FALLBACK_EXPLANATION = (
    "AI explanation is currently unavailable. The detection results are AI-generated, "
    "may include false positives or false negatives, are not a diagnosis, and should be reviewed "
    "by a licensed dental professional."
)

FINDING_LABELS = {
    "filling": "fillings",
    "impacted tooth": "possible impacted teeth",
    "crown": "crowns",
    "implant": "implant-like findings",
    "root canal treatment": "root canal treatment findings",
    "root piece": "possible root piece findings",
}

FINDING_CONTEXT = {
    "filling": "previously restored tooth areas",
    "impacted tooth": "teeth that may not have fully erupted",
    "crown": "cap-like restorations placed over teeth",
    "implant": "artificial tooth root-like structures",
    "root canal treatment": "signs of previous endodontic treatment",
    "root piece": "remaining root-like fragment appearances",
}

TURKISH_FINDING_LABELS = {
    "filling": "dolgu bulgusu",
    "impacted tooth": "gömülü diş benzeri bulgu",
    "crown": "kron benzeri bulgu",
    "implant": "implant benzeri bulgu",
    "root canal treatment": "kanal tedavisi benzeri bulgu",
    "root piece": "kök parçası benzeri bulgu",
}

TURKISH_FINDING_CONTEXT = {
    "filling": "genellikle daha önce restore edilmiş bir diş alanını ifade eder",
    "impacted tooth": "genellikle beklenen konumuna tam sürememiş bir dişi ifade eder",
    "crown": "genellikle dişin üzerine yerleştirilen kaplama benzeri bir restorasyonu ifade eder",
    "implant": "genellikle yapay diş kökü benzeri bir yapıyı ifade eder",
    "root canal treatment": "genellikle önceki endodontik tedavi izlerini ifade eder",
    "root piece": "genellikle kök benzeri kalan bir parça görünümünü ifade eder",
}

FINDING_SYNONYMS = {
    "filling": ("filling", "fillings", "dolgu"),
    "impacted tooth": ("impacted", "impacted tooth", "gömülü", "gomulu"),
    "crown": ("crown", "crowns", "kron", "kaplama"),
    "implant": ("implant", "implants"),
    "root canal treatment": ("root canal", "endodontic", "kanal tedavisi"),
    "root piece": ("root piece", "root fragment", "kök parçası", "kok parcasi"),
}

OVERVIEW_QUESTION_MARKERS = (
    "all findings",
    "all results",
    "what did you find",
    "what are the findings",
    "summary",
    "summarize",
    "everything",
    "tüm bulgular",
    "tum bulgular",
    "özet",
    "ozet",
)

LOCATION_QUESTION_MARKERS = (
    "which tooth",
    "which teeth",
    "where",
    "tooth number",
    "what tooth",
    "hangi diş",
    "hangi dis",
    "nerede",
    "kaç numara",
    "kac numara",
)

ADVICE_QUESTION_MARKERS = (
    "what should i do",
    "what do you think i should do",
    "what can i do",
    "ne yapmalıyım",
    "ne yapmaliyim",
    "ne yapmam",
)

TURKISH_SAFETY_SENTENCE = (
    "AI sonuçlarında yanlış pozitif veya yanlış negatif olabilir; bu bir tanı değildir ve diş hekimi tarafından incelenmelidir."
)

FORBIDDEN_PHRASES = (
    "the patient has",
    "requires treatment",
    "must be removed",
    "is infected",
    "is dangerous",
    "you need surgery",
    "you need medication",
    "needs surgery",
    "needs medication",
    "treatment plan",
    "prescribe",
    "likely accurate",
    "confidence level is high",
    "suggesting this finding is likely",
    "remember,",
    "should not replace",
)

VAGUE_FORBIDDEN_PHRASES = (
    "and others",
    "possibly more",
    "guidance",
    "further steps",
    "upper front",
    "lower front",
    "same region",
)

NUMBER_WORDS = (
    "thirty-one",
    "thirtieth",
    "nineteen",
    "seventeen",
    "sixteen",
    "fifteen",
    "fourteen",
    "thirteen",
    "twelve",
    "eleven",
)


def generate_xray_explanation(matched_findings: list[dict[str, Any]]) -> str:
    """
    Generate one safe patient-friendly explanation.

    Important design:
    - A deterministic explanation is always prepared first.
    - Ollama is used only to polish the wording.
    - If Ollama fails, times out, returns invalid text, or violates validation,
      the deterministic explanation is returned instead of "unavailable".
    """
    safe_findings = _prepare_findings(matched_findings)

    if not safe_findings:
        return NO_FINDINGS_EXPLANATION

    grouped = _group_findings(safe_findings)
    guaranteed_explanation = _build_template_explanation(grouped)

    if not _llm_explanation_enabled():
        return guaranteed_explanation

    if requests is None:
        logger.warning("Ollama explanation skipped because requests is unavailable.")
        return guaranteed_explanation

    prompts = [
        _build_polish_prompt(guaranteed_explanation),
        _build_compact_prompt(grouped),
    ]

    for index, prompt in enumerate(prompts, start=1):
        try:
            raw_text = _call_ollama(
                prompt=prompt,
                num_predict=180 if index == 1 else 140,
            )

            cleaned = _safe_paragraph(raw_text, grouped)

            if cleaned:
                return cleaned

            logger.warning("Ollama explanation attempt %s rejected by validation.", index)

        except Exception as exc:
            logger.warning("Ollama explanation attempt %s failed: %s", index, exc)

    logger.warning("Ollama failed after retries; returning deterministic explanation.")
    return guaranteed_explanation


def generate_chat_response(
    question: str,
    analysis_context: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Answer user questions as a safe chatbot.

    The chatbot only receives structured computer vision outputs and user text.
    It never receives image bytes and must not diagnose, prescribe, or invent findings.
    """
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        return "Please ask a question about the AI-generated dental X-ray results."

    if not _llm_explanation_enabled():
        return (
            "The chatbot is currently disabled. The detection results are AI-generated, "
            "may include false positives or false negatives, and are not a diagnosis."
        )

    if requests is None:
        logger.warning("Ollama chat skipped because requests is unavailable.")
        return _build_rule_based_chat_response(cleaned_question, analysis_context)

    try:
        messages = _build_chat_messages(cleaned_question, analysis_context, history)
        raw_text = _call_ollama_chat(messages)
        cleaned = _safe_chat_answer(raw_text, cleaned_question)
        if cleaned:
            return cleaned

        logger.warning("Ollama chat response rejected by validation.")
    except Exception as exc:
        logger.warning("Ollama chat failed: %s", exc)

    return _build_rule_based_chat_response(cleaned_question, analysis_context)


def _call_ollama(prompt: str, num_predict: int) -> str:
    if requests is None:
        raise RuntimeError("requests package is unavailable.")

    payload = {
        "model": _ollama_model(),
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.3,
            "num_predict": num_predict,
        },
    }

    response = requests.post(
        f"{_ollama_base_url()}/api/generate",
        json=payload,
        timeout=DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    explanation = data.get("response") if isinstance(data, dict) else None

    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Ollama returned an invalid explanation payload.")

    return explanation.strip()


def _call_ollama_chat(messages: list[dict[str, str]]) -> str:
    if requests is None:
        raise RuntimeError("requests package is unavailable.")

    payload = {
        "model": _ollama_model(),
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.5,
            "num_predict": 90,
        },
    }

    response = requests.post(
        f"{_ollama_base_url()}/api/chat",
        json=payload,
        timeout=DEFAULT_CHAT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an invalid chat payload.")

    return content.strip()


def _build_chat_messages(
    question: str,
    analysis_context: dict[str, Any] | None,
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    context = _sanitize_chat_context(analysis_context)
    recent_history = _sanitize_history(history)

    system_prompt = """
You are DentAI Chat, a cautious patient-friendly chatbot for a dental X-ray assistant.
You must not analyze images directly and you must not claim to see the uploaded X-ray.
Use only the structured computer vision outputs in the provided context and general dental education.
If the user asks about findings, mention only findings and tooth numbers present in the context.
Answer the user's specific question only. Do not summarize every finding unless the user explicitly asks for all findings, all results, or a summary.
If the user asks what one finding means, explain only that finding and do not list unrelated findings.
If the user asks what they should do, say that a licensed dentist or dental professional should review it; do not suggest a treatment plan.
If the context is empty or the answer is not supported by the context, say that the AI output does not provide enough information.
Do not diagnose, prescribe treatment, recommend medication, or make definitive clinical claims.
Do not say a finding is likely accurate, and do not mention confidence scores unless the user specifically asks about confidence.
Use cautious wording such as "may indicate", "could represent", "commonly refers to", and "may deserve professional review".
Do not repeat the safety disclaimer in every answer; the interface already shows it.
Answer in the same language as the user's question when possible.
Keep answers very short: one or two direct sentences.
Do not give long explanations unless the user asks for details.
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Structured computer vision context. This is not image data:\n"
                f"{_to_json_text(context)}"
            ),
        },
    ]

    messages.extend(recent_history)
    messages.append({"role": "user", "content": question})
    return messages


def _sanitize_chat_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {
            "matches": [],
            "findings": [],
            "ai_explanation": "",
            "matching_method": "YOLO segmentation + mask overlap",
        }

    matches = context.get("matched_results") or context.get("matches") or []
    findings = context.get("findings") or []

    return {
        "matches": _sanitize_match_items(matches),
        "findings": _sanitize_finding_items(findings),
        "ai_explanation": str(context.get("ai_explanation") or "")[:1200],
        "warnings": _sanitize_text_list(context.get("warnings") or [], limit=5),
        "matching_method": "YOLO segmentation + mask overlap",
    }


def _sanitize_match_items(items: Any) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []

    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue

        sanitized_item = {
            "tooth_number": item.get("tooth_number", item.get("matched_tooth_number")),
            "finding": item.get("finding", item.get("finding_class")),
        }

        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            sanitized_item["confidence"] = round(float(confidence), 4)

        overlap_score = item.get("overlap_score")
        if isinstance(overlap_score, (int, float)):
            sanitized_item["overlap_score"] = round(float(overlap_score), 4)

        if sanitized_item["finding"]:
            sanitized.append(sanitized_item)

    return sanitized[:40]


def _sanitize_finding_items(items: Any) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []

    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue

        sanitized_item = {
            "finding_name": item.get("finding_name", item.get("finding")),
        }

        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            sanitized_item["confidence"] = round(float(confidence), 4)

        if sanitized_item["finding_name"]:
            sanitized.append(sanitized_item)

    return sanitized[:40]


def _sanitize_text_list(items: Any, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []

    return [str(item)[:300] for item in items[:limit] if item]


def _sanitize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []

    for item in history or []:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = str(item.get("content") or "").strip()

        if role not in {"user", "assistant"} or not content:
            continue

        sanitized.append({"role": role, "content": content[:800]})

    return sanitized[-6:]


def _safe_chat_answer(text: str, question: str) -> str | None:
    answer = _paragraphize(text)

    if not answer:
        return None

    if _contains_forbidden_language(answer):
        return None

    if _sentence_count(answer) > 2:
        return None

    if _mentions_unasked_findings(answer, question):
        return None

    return _limit_sentences(answer, max_sentences=2)


def _build_rule_based_chat_response(question: str, analysis_context: dict[str, Any] | None) -> str:
    context = _sanitize_chat_context(analysis_context)
    matches = context.get("matches", [])
    wants_turkish = _wants_turkish(question)
    mentioned_findings = _mentioned_findings(question)
    asks_for_overview = _has_any_marker(question, OVERVIEW_QUESTION_MARKERS)
    asks_for_location = _has_any_marker(question, LOCATION_QUESTION_MARKERS)
    asks_for_advice = _has_any_marker(question, ADVICE_QUESTION_MARKERS)

    if matches:
        grouped: dict[str, list[str]] = {}
        for item in matches:
            finding = str(item.get("finding") or "").strip().lower()
            tooth = item.get("tooth_number")
            if finding and tooth is not None:
                grouped.setdefault(finding, []).append(str(tooth))

        if mentioned_findings:
            grouped = {finding: teeth for finding, teeth in grouped.items() if finding in mentioned_findings}
        elif not asks_for_overview:
            if asks_for_advice:
                return _advice_response(wants_turkish)
            return _ask_for_specific_question_response(wants_turkish)

        if not grouped:
            return _general_finding_response(question, wants_turkish, asks_for_advice)

        finding_parts = []
        context_parts = []
        for finding, teeth in grouped.items():
            unique_teeth = sorted(set(teeth), key=lambda value: int(value) if value.isdigit() else value)
            if wants_turkish:
                label = TURKISH_FINDING_LABELS.get(finding, f"{finding} bulgusu")
                if asks_for_location or asks_for_overview:
                    finding_parts.append(f"{_format_tooth_list(unique_teeth)} numaralı dişte {label}")
                if finding in TURKISH_FINDING_CONTEXT:
                    context_parts.append(f"{label} {TURKISH_FINDING_CONTEXT[finding]}")
            else:
                label = FINDING_LABELS.get(finding, f"{finding} findings")
                if asks_for_location or asks_for_overview:
                    finding_parts.append(f"{label} on tooth {_format_tooth_list(unique_teeth)}")
                if finding in FINDING_CONTEXT:
                    context_parts.append(f"{label.capitalize()} commonly refer to {FINDING_CONTEXT[finding]}")

        if wants_turkish:
            if finding_parts:
                answer = f"AI sonucu {_join_phrases(finding_parts)} gösteriyor"
            else:
                answer = ""
            if context_parts and answer:
                answer = f"{answer}; {_join_phrases(context_parts)}."
            elif context_parts:
                answer = f"{_join_phrases(context_parts)}."
            else:
                answer = f"{answer}."
            return _append_advice_if_needed(answer, asks_for_advice, wants_turkish)

        if finding_parts:
            answer = f"The AI results show {_join_phrases(finding_parts)}"
        else:
            answer = ""
        if context_parts and answer:
            answer = f"{answer}; {_join_phrases(context_parts)}."
        elif context_parts:
            answer = f"{_join_phrases(context_parts)}."
        else:
            answer = f"{answer}."
        return _append_advice_if_needed(answer, asks_for_advice, wants_turkish)

    general_response = _general_finding_response(question, wants_turkish, asks_for_advice)
    if general_response:
        return general_response

    return CHAT_FALLBACK_RESPONSE


def _to_json_text(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=True, indent=2)


def _limit_sentences(text: str, max_sentences: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if len(sentences) <= max_sentences:
        return text
    return " ".join(sentences[:max_sentences])


def _sentence_count(text: str) -> int:
    return len([sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence.strip()])


def _mentioned_findings(question: str) -> set[str]:
    lower_question = question.lower()
    mentioned = set()

    for finding, synonyms in FINDING_SYNONYMS.items():
        if any(synonym in lower_question for synonym in synonyms):
            mentioned.add(finding)

    return mentioned


def _mentions_unasked_findings(answer: str, question: str) -> bool:
    mentioned = _mentioned_findings(question)
    asks_for_overview = _has_any_marker(question, OVERVIEW_QUESTION_MARKERS)
    lower_answer = answer.lower()

    if not mentioned:
        if asks_for_overview:
            return False
        return any(
            synonym in lower_answer
            for synonyms in FINDING_SYNONYMS.values()
            for synonym in synonyms
        )

    for finding, synonyms in FINDING_SYNONYMS.items():
        if finding in mentioned:
            continue
        if any(synonym in lower_answer for synonym in synonyms):
            return True

    return False


def _has_any_marker(question: str, markers: tuple[str, ...]) -> bool:
    lower_question = question.lower()
    return any(marker in lower_question for marker in markers)


def _general_finding_response(question: str, wants_turkish: bool, asks_for_advice: bool) -> str | None:
    mentioned = _mentioned_findings(question)

    if not mentioned:
        if asks_for_advice:
            return _advice_response(wants_turkish)
        return None

    finding = sorted(mentioned)[0]
    if wants_turkish:
        label = TURKISH_FINDING_LABELS.get(finding, finding)
        context_text = TURKISH_FINDING_CONTEXT.get(finding)
        if not context_text:
            return _advice_response(wants_turkish) if asks_for_advice else None
        answer = f"Genel olarak {label} {context_text}."
        return _append_advice_if_needed(answer, asks_for_advice, wants_turkish)

    label = FINDING_LABELS.get(finding, finding)
    context_text = FINDING_CONTEXT.get(finding)
    if not context_text:
        return _advice_response(wants_turkish) if asks_for_advice else None
    answer = f"In general, {label} commonly refer to {context_text}."
    return _append_advice_if_needed(answer, asks_for_advice, wants_turkish)


def _advice_response(wants_turkish: bool) -> str:
    if wants_turkish:
        return "Bu sonuca göre kesin bir işlem söyleyemem; en doğru adım röntgeni bir diş hekimine göstermektir."
    return "I cannot decide treatment from this AI result; the best next step is to have a dentist review the X-ray."


def _ask_for_specific_question_response(wants_turkish: bool) -> str:
    if wants_turkish:
        return "Belirli bir bulguyu sorarsan onu kısaca açıklayabilirim."
    return "Ask about a specific finding and I can explain it briefly."


def _append_advice_if_needed(answer: str, asks_for_advice: bool, wants_turkish: bool) -> str:
    if not asks_for_advice:
        return answer

    advice = _advice_response(wants_turkish)
    if answer.strip().endswith(advice):
        return answer
    return f"{answer} {advice}"


def _wants_turkish(question: str) -> bool:
    lower_question = question.lower()
    turkish_markers = ("ı", "ğ", "ü", "ş", "ö", "ç", "ne demek", "nedir", "anlama", "diş", "dolgu")
    return any(marker in lower_question for marker in turkish_markers)


def _prepare_findings(matched_findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []

    for item in matched_findings or []:
        finding_name = str(item.get("finding") or item.get("finding_class") or "").strip().lower()
        tooth_number = item.get("tooth_number", item.get("matched_tooth_number"))

        if not finding_name or tooth_number is None:
            continue

        prepared.append(
            {
                "finding": finding_name,
                "tooth_number": str(tooth_number).strip(),
            }
        )

    return prepared


def _group_findings(findings: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}

    for item in findings:
        finding = str(item.get("finding", "")).strip().lower()
        tooth = str(item.get("tooth_number", "")).strip()

        if not finding or not tooth:
            continue

        grouped.setdefault(finding, set()).add(tooth)

    cleaned: dict[str, list[str]] = {}

    for finding, teeth in grouped.items():
        cleaned[finding] = sorted(
            teeth,
            key=lambda value: int(value) if value.isdigit() else value,
        )

    return cleaned


def _build_template_explanation(grouped: dict[str, list[str]]) -> str:
    finding_parts: list[str] = []
    context_parts: list[str] = []

    for finding, teeth in grouped.items():
        label = FINDING_LABELS.get(finding, f"{finding} findings")
        teeth_text = _format_tooth_list(teeth)

        finding_parts.append(f"{label} on teeth {teeth_text}")

        context = FINDING_CONTEXT.get(finding)
        if context:
            context_parts.append(f"{label.capitalize()} commonly refer to {context}")

    if not finding_parts:
        return NO_FINDINGS_EXPLANATION

    summary_sentence = f"The AI analysis identified {_join_phrases(finding_parts)}."

    if context_parts:
        context_sentence = f"{_join_phrases(context_parts)}."
        return f"{summary_sentence} {context_sentence} {SAFETY_SENTENCE}"

    return f"{summary_sentence} {SAFETY_SENTENCE}"


def _build_polish_prompt(template_text: str) -> str:
    return f"""
Rewrite this dental AI explanation into one clear, natural English paragraph.

Text:
{template_text}

Strict rules:
- Keep all tooth numbers exactly as numeric digits.
- Do not remove any tooth numbers.
- Do not add new findings.
- Do not infer anatomical regions.
- Do not mention confidence scores.
- Do not mention treatment or medication.
- Do not use phrases like "and others", "possibly more", "guidance", or "further steps".
- Do not use titles, bullets, labels, or markdown.
- Keep the meaning unchanged.
- Keep it concise and patient-friendly.
- Return only the final paragraph.
""".strip()


def _build_compact_prompt(grouped: dict[str, list[str]]) -> str:
    compact_lines = []

    for finding, teeth in grouped.items():
        label = FINDING_LABELS.get(finding, f"{finding} findings")
        compact_lines.append(f"{label}: {', '.join(teeth)}")

    return f"""
Create one short English paragraph from these dental AI findings:

{'; '.join(compact_lines)}

Rules:
- Use numeric tooth numbers only.
- Group similar findings together.
- Do not add new findings.
- Do not infer tooth positions or anatomical regions.
- Do not mention treatment, medication, or confidence scores.
- Do not use phrases like "and others", "possibly more", "guidance", or "further steps".
- End with this exact sentence: {SAFETY_SENTENCE}
- Return only the final paragraph.
""".strip()


def _safe_paragraph(text: str, grouped: dict[str, list[str]]) -> str | None:
    paragraph = _paragraphize(text)

    if not paragraph:
        return None

    if _contains_forbidden_language(paragraph):
        return None

    if _contains_vague_forbidden_phrases(paragraph):
        return None

    if _spells_tooth_numbers_as_words(paragraph):
        return None

    if not _contains_expected_tooth_numbers(paragraph, grouped):
        return None

    return _ensure_safety_sentence(paragraph)


def _paragraphize(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]+|\d+[\).])\s*", "", line).strip()
        line = re.sub(r"^Sentence\s*\d+\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        line = re.sub(r"^AI\s+Overview\s*:?\s*", "", line, flags=re.IGNORECASE).strip()

        if line:
            cleaned_lines.append(line)

    paragraph = " ".join(cleaned_lines)
    paragraph = paragraph.replace("**", "").replace("__", "")
    paragraph = re.sub(r"\s+", " ", paragraph).strip()

    return paragraph


def _contains_forbidden_language(text: str) -> bool:
    lower_text = text.lower()
    return any(phrase in lower_text for phrase in FORBIDDEN_PHRASES)


def _contains_vague_forbidden_phrases(text: str) -> bool:
    lower_text = text.lower()
    return any(phrase in lower_text for phrase in VAGUE_FORBIDDEN_PHRASES)


def _spells_tooth_numbers_as_words(text: str) -> bool:
    lower_text = text.lower()
    return any(word in lower_text for word in NUMBER_WORDS)


def _contains_expected_tooth_numbers(text: str, grouped: dict[str, list[str]]) -> bool:
    for teeth in grouped.values():
        for tooth in teeth:
            if re.search(rf"\b{re.escape(tooth)}\b", text) is None:
                return False

    return True


def _ensure_safety_sentence(text: str) -> str:
    lower_text = text.lower()

    has_safety = (
        "ai-generated" in lower_text
        and "false positives" in lower_text
        and "false negatives" in lower_text
        and "not a diagnosis" in lower_text
        and "licensed dental professional" in lower_text
    )

    if has_safety:
        return text

    return f"{text} {SAFETY_SENTENCE}"


def _format_tooth_list(teeth: list[str]) -> str:
    if not teeth:
        return ""

    if len(teeth) == 1:
        return teeth[0]

    if len(teeth) == 2:
        return f"{teeth[0]} and {teeth[1]}"

    return ", ".join(teeth[:-1]) + f", and {teeth[-1]}"


def _join_phrases(parts: list[str]) -> str:
    if not parts:
        return ""

    if len(parts) == 1:
        return parts[0]

    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"

    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _llm_explanation_enabled() -> bool:
    value = os.getenv("ENABLE_LLM_EXPLANATION", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
