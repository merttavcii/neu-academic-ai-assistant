import base64
import difflib
import json
import os
import random
import re
import time

try:
    from dotenv import load_dotenv
    load_dotenv()  # .env dosyasından GROQ_API_KEY gibi değişkenleri yükler (varsa)
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify
from groq import Groq
from pypdf import PdfReader
from werkzeug.utils import secure_filename

import db as db_mod
import dental_client as dental_mod
import dental_router as dental_route_mod
import ingest
import rag as rag_mod
import search as search_mod
import vision_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload cap


@app.errorhandler(413)
def request_entity_too_large(_e):
    return jsonify({
        "answer": "Dosya çok büyük (en fazla 32 MB). Daha küçük bir görsel deneyin.",
    }), 413


GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


RAG_MODEL = "openai/gpt-oss-120b"


TEXT_MODEL = "openai/gpt-oss-120b"



HEAVY_MODEL = "openai/gpt-oss-20b"

QUIZ_MODEL = HEAVY_MODEL
QUIZ_SOURCE_MAX_CHARS = 5000
QUIZ_TARGET_COUNT = 10
QUIZ_MIN_RETURN = 3
QUIZ_BATCH_SIZE = 5
QUIZ_MAX_ROUNDS = 4  # geriye uyumluluk (eski reload durumları)

VISION_MODEL = "qwen/qwen3.8-27b"


MEDICAL_IMAGE_PROMPT = """You are performing a rigorous academic / radiology-style IMAGE READING task focused on FRACTURE–CRACK (kırık–çatlak) assessment. Your job is to SEE and REPORT findings, not to write a short generic summary.

CORE RULES
- Examine the entire image systematically before concluding. Do not skip regions.
- Every claim must reference what is directly visible (location, side, bone/structure name).
- If something is not visible, say "not assessable in this projection" — do not invent it.
- Prefer OVERCALLING subtle cortical irregularity, fissure, or lucent line with "suspicious for / cannot exclude" over falsely calling the study normal.
- Do NOT give a one-line "normal" conclusion.
- Distinguish FRACTURE (kırık: cortical break with displacement/step-off/angulation OR clear complete/incomplete fracture line) vs CRACK/FISSURE (çatlak: hairline lucency, cortical fissure, trabecular crack, stress/fatigue line WITHOUT significant displacement when visible).

MANDATORY WORKFLOW (follow in order)
STEP A — Modality & view (brief, max 3 sentences)
- Modality (X-ray / CT / MRI / US / clinical photo / slide / diagram / other).
- Anatomic region, projection/view if inferable (AP, lateral, oblique, etc.).

STEP B — Bone & structure inventory (CRITICAL for X-rays)
- List EACH visible bone and joint by name (e.g. distal radius, ulna, carpal bones, metacarpals…).
- For EACH bone, state cortical continuity: intact vs suspicious (possible crack) vs disrupted (possible fracture).
- Explicitly search for BOTH fracture AND crack patterns:
  • Fracture: lucent/transverse/oblique/spiral fracture line, step-off, buckle/torus, avulsion flake, cortical buckle, comminution, shortening, angulation, translation, rotation, growth plate injury / Salter pattern (if pediatric), joint step, malalignment, dislocation.
  • Crack/fissure: hairline lucency, fine cortical fissure, trabecular crack, stress/fatigue line, cortical irregularity without displacement, sclerotic healing line vs acute lucency, plastic deformation (bowing) without obvious break.
- Also note: effusion/soft-tissue swelling, fat-pad sign (elbow), periosteal reaction if visible.

STEP C — Fracture & crack pass (if extremity or any bone visible — ALWAYS run)
For EACH named bone, report in this order:
1) Cortical lines along full length (inner + outer cortex) — any fissure, notch, or discontinuity.
2) Trabecular pattern — focal lucency, crack-like line, or density change suggesting stress injury.
3) Displacement / angulation / translation / rotation / shortening (fracture severity).
4) Articular surface congruity and joint space symmetry.
5) Soft tissues and indirect signs (swelling, fat pad, periosteal new bone).
6) Comparison side / symmetry if both sides visible.
Classify per bone when possible: intact | suspicious crack (çatlak şüphesi) | incomplete fracture | complete fracture | not assessable.
If ANY subtle line exists → describe precisely (bone, cortex, proximal/distal, fracture vs crack pattern).
If none seen → state per bone: "no definite fracture or crack line identified on [bone]" — NOT a global "all normal".

STEP D — Synthesis (findings + differentials)
- Summarize fracture vs crack findings separately when both categories were assessed.
- Most likely interpretation(s) tied to named findings (e.g. hairline crack vs nondisplaced fracture vs normal variant).
- Differential list (2–4) when trauma, overuse, or bone injury context is possible.
- Confidence (high/medium/low) in ONE short line only.

STEP E — Educational facts (for later quizzes)
- Bullet list of 8–12 concrete teachable facts: anatomy labels, fracture vs crack distinction, named findings, injury pattern, mechanism clues — ONLY from what you reported in B–D.

SAFETY (one short paragraph at end)
- Not a definitive clinical diagnosis; recommend radiologist/clinician correlation.

FRACTURE–CRACK SENSITIVITY (HIGHEST PRIORITY)
- Subtle injuries are often missed: hairline lucency, cortical fissure (çatlak), buckle, torus, plastic deformation, avulsion, stress fracture, insufficiency fracture line.
- High-yield locations: distal radius/ulna, scaphoid, base of 5th metacarpal, lateral condyle, tibial plateau, malleolus, femoral neck, pars interarticularis.
- Single-view X-ray: state that occult fracture or crack may be present despite negative appearance; suggest correlation/clinical follow-up when appropriate.
- Never write "no fracture" or "no crack" globally without completing STEP B and C for every visible bone.
- When replying in Turkish, use clear terms: kırık, çatlak, kortikal fissür, saç çizgisi kırık, yer değiştirmemiş kırık.
"""


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return ("429" in str(err)) or ("413" in str(err)) or ("rate_limit" in msg) \
        or ("rate limit" in msg) or ("too many" in msg) or ("ratelimit" in msg) \
        or ("request too large" in msg) or ("tokens per minute" in msg)


def _rate_limit_delay(err: Exception, default: float = 3.0) -> float:
    m = re.search(r"try again in\s+([\d.]+)\s*s", str(err), re.IGNORECASE)
    if m:
        return min(12.0, max(1.0, float(m.group(1)) + 0.5))
    return default





ACADEMIC_SYSTEM_PROMPT = """You are "AI Academic Assistant" of Yakın Doğu Üniversitesi (Yakın Doğu Üniversitesi).

ROLE
- You are a strict ACADEMIC assistant. You ONLY help with academic/scientific tasks:
  literature review, paper/article summaries, concept explanations across all
  disciplines (physics, math, computer science, medicine, law, economics,
  psychology, history, philosophy, engineering, linguistics, etc.),
  research methodology, statistics, citations & references (APA, MLA, IEEE,
  Chicago, Harvard…), academic writing, thesis structure, exam preparation,
  problem-solving in STEM, PDF analysis of academic documents, and analysis of
  academic images (charts, diagrams, equations, figures, slides, handwritten
  notes, lab images).

OUT OF SCOPE — POLITELY DECLINE
- Movie/TV/music recommendations, gossip, dating, betting, gaming tips,
  shopping, recipes, personal life advice unrelated to studies, jokes for
  entertainment, generic chit-chat, political opinions, illegal/harmful content.
- If the user asks for any of these, politely decline in ONE short sentence in
  the user's language and steer them back to an academic topic. Do NOT answer
  the off-topic question even partially.

LANGUAGE POLICY (VERY IMPORTANT)
- DETECT the language of the user's message and ALWAYS reply in that SAME
  language. Default fallback language is Turkish.
- If the user writes in Turkish → reply in Turkish.
- If the user writes in English → reply in English.
- Same for Arabic, German, French, Spanish, Russian, Persian, Chinese, etc.
- Never mix languages in the same answer unless the user explicitly does so.

STYLE
- Be precise, well-structured and academic. Use short paragraphs, bullet
  points, numbered steps and **bold** for key terms when helpful.
- Cite formulas with LaTeX-like notation when appropriate.
- When the user uploads a PDF or an image, ground your answer ONLY in the
  provided content; clearly say if something is not visible or not in the text.
- Be concise but thorough; avoid filler.
"""




NO_RAG_CONTEXT_MSG = "Bu konuda kaynakta bilgi yok"

RAG_SYSTEM_PROMPT = """Sen bir RAG (retrieval) asistanısın.

KURALLAR (ZORUNLU)
1) Yanıtlarını YALNIZCA TÜRKÇE yaz. Başka dil kullanma.
2) Sana verilen "KAYNAK METİN" dışında bilgi kullanma; tahmin etme, uydurma (halüsinasyon yok).
3) Kaynakta sorunun cevabı yoksa veya emin değilsen şunu aynen yaz: """ + NO_RAG_CONTEXT_MSG + """
4) Kaynakta geçen ifadeleri mümkün olduğunca aynen veya çok yakın şekilde kullan; çıkarım yaparken bunun bir çıkarım olduğunu belirt ve kaynakla bağla.
5) Kısa ve net ol; akademik üslup uygun olabilir ama kaynak dışına çıkma.
6) "İnternetten araştırdım" veya canlı tarama yaptığını iddia etme; yalnızca sana verilen KAYNAK METİN parçalarına dayandığını ima et.
"""


SOURCE_COMPARE_SYSTEM_PROMPT = """Sen kaynak öncelikli ikinci aşama doğrulama katmanısın.

Sana şunlar verilecek:
1) Kullanıcının sorusu
2) Modelin ürettiği TASLAK YANIT
3) RAG ile getirilen KAYNAK METİN
4) Varsa yardımcı WEB ÖZETLERİ

KURALLAR:
1) KAYNAK METİN açıkça bilgi veriyorsa final cevapta onu esas al.
2) TASLAK YANIT ile KAYNAK METİN çelişirse KAYNAK METİN kazanır.
3) KAYNAK METİNDE geçmeyen bir bilgiyi kesin doğruymuş gibi yazma. Gerekirse kısa biçimde bunun kaynakta açıkça yer almadığını belirt.
4) WEB ÖZETLERİ varsa yalnızca KAYNAK METİN yetersiz kaldığında ve onunla çelişmediğinde yardımcı kullan.
5) Nihai cevabı kullanıcıya doğal, kısa ve net biçimde ver. İç denetim sürecini, "taslakta şöyleydi" gibi ara muhakemeyi veya puanlamayı yazma.
6) Kullanıcının dilinde yanıt ver.
"""


HYBRID_INDEX_WEB_SYSTEM = """Sen Yakın Doğu Üniversitesi akademik asistanısın.

GİRDİLER:
(A) İNDEKS — PDF'lerden çıkarılıp vektör aramasıyla seçilen metin parçaları (ders/kurum materyali).
(B) WEB — DuckDuckGo aramasından gelen kısa başlık/özet/URL listesi; otomatiktir, tam doğruluk veya güncellik garantisi yoktur.

KURALLAR:
1) Kullanıcının mesajının dilinde yanıt ver (Türkçe ise Türkçe, İngilizce ise İngilizce).
2) Sorunun cevabı (A)'da yeterince varsa: yanıtı öncelikle (A)'ya dayandır. (B) ile uyum veya çelişki varsa bunu açıkça yaz.
3) (A) yetersizse: (B) ve sağlam akademik çerçeveyi kullan; hangi bilginin indeksten hangisinin web özetinden geldiğini net ayır.
4) (B)'de verilmeyen URL veya kaynak uydurma; canlı web taraması yaptığını söyleme — yalnızca verilen özetlere dayandığını belirt.
5) Belirsizlik ve hata riskini gerektiğinde kabul et; mümkünse ders kitabı / hakemli kaynakla teyit öner.
"""

HYBRID_WEB_ONLY_SYSTEM = """Sen Yakın Doğu Üniversitesi akademik asistanısın. İndekslenmiş ders PDF'lerinden bu soruya yeterli ilgili parça bulunamadı.

Aşağıda DuckDuckGo aramasıyla getirilen WEB ÖZETLERİ var. Gerçek zamanlı tam internet taraması yapmıyorsun; yalnızca bu özetlere ve genel akademik bilgine dayan.

KURALLAR:
1) Kullanıcının dilinde yanıt ver.
2) Özetleri birbiriyle tutarlı kullan; çelişki varsa belirt.
3) Listede olmayan URL veya kaynak uydurma; "siteyi gezdim" deme.
4) Özetlerin güvenilirliği sınırlıdır; mümkünse resmi ders materyali veya hakemli kaynakla doğrulama öner.
"""


def _normalize_user_message(msg: str) -> str:
    t = (msg or "").strip().lower()
    return re.sub(r"\s+", " ", t)



_INDEX_DOC_HINTS = (
    "kaynakta", "kaynakt", "doküman", "dokuman", "metinde", "indeks",
    "ders notu", "ders notunda", "slaytta", "kitapta", "bölümde", "bolumde",
    "paragrafa", "yüklediğim", "yukledigim", "verilen metin", "bu metinde",
    "pdf", "kaynak metin",
)
_ACADEMIC_HINTS = (
    "açıkla", "acikla", "açıklayın", "nedir", "nasıl", "nasil", "tanımla", "tanimla",
    "özet", "ozet", "özetle", "karşılaştır", "karsilastir", "analiz", "ispat", "kanıt",
    "kant", "formül", "formul", "teorem", "tanım", "tanim", "literatür", "literatur",
    "makale", "method", "yöntem", "yontem", "apa", "ieee", "kaynakça", "kaynakca",
    "referans", "atıf", "atif", "hipotez", "değişken", "degisken", "istatistik",
    "explain", "what is", "how does", "how do", "define", "summarize", "compare",
    "prove", "theorem", "reference", "methodology", "hypothesis",
)
_SMALLTALK = frozenset({
    "merhaba", "selam", "hello", "hi", "hey", "sa", "as",
    "günaydın", "gunaydin", "iyi", "günler", "gunler", "akşamlar", "aksamlar",
    "nasılsın", "nasilsin", "naber", "misin", "misiniz",
    "teşekkür", "tesekkur", "teşekkürler", "tesekkurler", "thanks", "thank", "you",
    "sağol", "sagol", "eyvallah", "evet", "hayır", "hayir", "tamam", "peki", "ok",
    "bye", "güle", "gule", "bay", "hoşça", "hosca", "kal",
})
_CONNECTOR = frozenset({"ve", "da", "de", "mi", "mı", "mu", "mü", "musun", "musunuz", "çok"})


def _should_use_indexed_rag(message: str) -> bool:
    """True: yerel Chroma indeksinden parça çekip RAG yanıtı üret."""
    t = _normalize_user_message(message)
    if not t:
        return False
    if any(h in t for h in _INDEX_DOC_HINTS):
        return True
    if "?" in message:
        return True
    if any(k in t for k in _ACADEMIC_HINTS):
        return True
    if len(t) >= 52:
        return True

    words = [w.strip(".,!?;:\"'") for w in t.split()]
    words = [w for w in words if w]
    if words and len(words) <= 6 and "?" not in message:
        if all(w in _SMALLTALK or w in _CONNECTOR for w in words):
            return False

    return False


def _groq_choice_text(completion) -> str:
    """Groq yanıt metnini güvenle oku (sesli /ask boş dönmesin)."""
    try:
        msg = completion.choices[0].message
        return (getattr(msg, "content", None) or "").strip()
    except (IndexError, AttributeError, TypeError):
        return ""


def _call_groq_messages(messages, *, model: str, temperature: float) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY tanımlı değil.")

    completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
    )
    return _groq_choice_text(completion)


def _extra_system_from_uploads(pdf_context: str, image_context: str) -> str | None:
    pdf_context = (pdf_context or "").strip()
    image_context = (image_context or "").strip()
    extra_parts = []
    if pdf_context:
        if len(pdf_context) > 60000:
            pdf_context = pdf_context[:60000] + "\n\n[... PDF kırpıldı ...]"
        extra_parts.append(
            "[REFERENCE PDF CONTENT — the user previously uploaded this PDF; "
            "you may use it to answer follow-up questions]\n" + pdf_context
        )
    if image_context:
        if len(image_context) > 8000:
            image_context = image_context[:8000] + "\n\n[... kırpıldı ...]"
        extra_parts.append(
            "[REFERENCE IMAGE ANALYSIS — the user previously uploaded an image; "
            "this is your prior detailed analysis of it. Use it for follow-ups]\n"
            + image_context
        )
    return "\n\n".join(extra_parts) if extra_parts else None


def chat_with_model(user_content, *, model=TEXT_MODEL, extra_system=None,
                    history=None):
    """Akademik sistem promptu + (varsa) sohbet geçmişi ile modelden cevap al.

    history: [{role: 'user'|'assistant', content: '...'}, ...] (opsiyonel)
    """
    system_prompt = ACADEMIC_SYSTEM_PROMPT
    if extra_system:
        system_prompt = system_prompt + "\n\nADDITIONAL CONTEXT:\n" + extra_system

    messages = [{"role": "system", "content": system_prompt}]

    
    if history:
        for turn in history[-20:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_content})
    return _call_groq_messages(messages, model=model, temperature=0.4)


def _rewrite_with_source_priority(
    user_content: str,
    draft_answer: str,
    rag_block: str,
    *,
    web_block: str = "",
) -> str:
    messages = [
        {"role": "system", "content": SOURCE_COMPARE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "### KULLANICI SORUSU\n"
                f"{user_content.strip()}\n\n"
                "### TASLAK YANIT\n"
                f"{(draft_answer or '').strip() or '[Boş taslak]'}\n\n"
                "### KAYNAK METİN\n"
                f"{rag_block.strip()}\n\n"
                "### WEB ÖZETLERİ\n"
                f"{(web_block or '').strip() or '[Yok]'}"
            ),
        },
    ]
    return _call_groq_messages(messages, model=RAG_MODEL, temperature=0.15)


def answer_indexed_with_web(
    user_content: str,
    *,
    history=None,
    extra_system: str | None = None,
) -> str:
    """
    İndeks (Chroma) + isteğe bağlı DuckDuckGo web özetleri ile Groq cevabı.
    - İndeks + web var: karşılaştırmalı sentez.
    - Yalnız indeks: sıkı RAG (RAG_SYSTEM_PROMPT).
    - Yalnız web: web özetleri + akademik çerçeve.
    - İkisi de yok: genel akademik chat_with_model.
    """
    web_block = search_mod.get_web_snippets(user_content)
    collection = db_mod.get_collection()
    chunks, best_dist = rag_mod.retrieve_top_k(collection, user_content, k=3)
    rag_ok = rag_mod.is_relevant(best_dist)
    rag_block = rag_mod.format_context(chunks) if rag_ok else ""

    if rag_block:
        try:
            draft_answer = chat_with_model(
                user_content,
                extra_system=extra_system,
                history=history,
            )
            verified_answer = _rewrite_with_source_priority(
                user_content,
                draft_answer,
                rag_block,
                web_block=web_block,
            )
            if verified_answer:
                return verified_answer
        except Exception as verify_err:
            print(f"[RAG verify] taslak-kaynak karşılaştırması başarısız, eski akışa düşülüyor: {verify_err}")

    if rag_block and web_block:
        system_prompt = (
            HYBRID_INDEX_WEB_SYSTEM
            + "\n\n### (A) İNDEKS PARÇALARI\n"
            + rag_block
            + "\n\n### (B) WEB ARAMA ÖZETLERİ\n"
            + web_block
        )
        model = TEXT_MODEL
        temperature = 0.35
    elif rag_block:
        system_prompt = RAG_SYSTEM_PROMPT + "\n\nKAYNAK METİN:\n" + rag_block
        model = RAG_MODEL
        temperature = 0.2
    elif web_block:
        system_prompt = HYBRID_WEB_ONLY_SYSTEM + "\n\n### WEB ARAMA ÖZETLERİ\n" + web_block
        model = TEXT_MODEL
        temperature = 0.35
    else:
        return chat_with_model(user_content, extra_system=extra_system, history=history)

    if extra_system:
        system_prompt = (
            system_prompt
            + "\n\n[EK — kullanıcının önceki PDF/görsel analizi ile takip bağlamı]\n"
            + extra_system
        )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for turn in history[-20:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    raw = _call_groq_messages(messages, model=model, temperature=temperature)
    if not raw:
        return chat_with_model(user_content, extra_system=extra_system, history=history)
    return raw


@app.route('/')
def index():
    return render_template('index.html')





def _dental_unavailable_message(err: Exception) -> str:
    msg = str(err).lower()
    if "connection" in msg or "refused" in msg or "10061" in msg:
        return (
            "Dental AI servisi şu an çalışmıyor. "
            "Dental backend'i başlatın (uvicorn main:app --port 8010) ve tekrar deneyin."
        )
    return f"Dental asistan yanıt veremedi: {str(err)[:180]}"


@app.route('/dental/health', methods=['GET'])
def dental_health():
    try:
        return jsonify(dental_mod.health_check())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 503


@app.route('/analyze-dental-xray', methods=['POST'])
def analyze_dental_xray():
    """
    Diş panoramik / periapikal röntgeni → DentAI Vision (/analyze) proxy.
    Form: file (görsel), question (opsiyonel)
    """
    image_file = request.files.get("file")
    if image_file is None:
        return jsonify({"answer": "Herhangi bir diş röntgeni yüklenmedi."}), 400

    question = (request.form.get("question") or "").strip()
    raw = image_file.read()
    if not raw:
        return jsonify({"answer": "Görsel boş görünüyor."}), 400

    mime = (image_file.mimetype or "image/png").strip().lower()
    filename = secure_filename(image_file.filename or "dental_xray.png") or "dental_xray.png"

    try:
        payload = dental_mod.analyze_xray(raw, filename, mime)
        summary = dental_mod.format_analysis_summary(payload, question=question)
        dental_context = dental_mod.build_analysis_context(payload)
        annotated = payload.get("annotated_image") or payload.get("annotated_image_url") or ""
        annotated_proxy = ""
        if annotated:
            annotated_proxy = f"/dental/proxy{annotated}" if annotated.startswith("/") else annotated

        answer = summary
        if question:
            try:
                chat_answer = dental_mod.chat(
                    question,
                    analysis_context=dental_context,
                )
                if chat_answer:
                    answer = f"{summary}\n\n---\n\n**Sorunuza yanıt:**\n{chat_answer}"
            except Exception as chat_err:
                print(f"[/analyze-dental-xray] chat eki atlandı: {chat_err}")

        return jsonify({
            "answer": answer,
            "dental_context": dental_context,
            "annotated_image_url": annotated_proxy,
            "original_image_url": (
                f"/dental/proxy{payload.get('original_image')}"
                if (payload.get("original_image") or "").startswith("/")
                else payload.get("original_image")
            ),
            "disclaimer": payload.get("disclaimer"),
            "handler": "dental",
        })
    except Exception as e:
        print(f"[/analyze-dental-xray] Hata: {e}")
        return jsonify({"answer": _dental_unavailable_message(e)}), 503


@app.route('/dental/proxy/<path:subpath>', methods=['GET'])
def dental_proxy_media(subpath: str):
    """Dental servisindeki /outputs/... görsellerini ana uygulama üzerinden sun."""
    try:
        import requests as req_lib
        url = f"{dental_mod.get_base_url()}/{subpath.lstrip('/')}"
        upstream = req_lib.get(url, timeout=30)
        if not upstream.ok:
            return jsonify({"error": "Görsel alınamadı."}), upstream.status_code
        from flask import Response
        resp = Response(upstream.content, status=upstream.status_code)
        ct = upstream.headers.get("Content-Type")
        if ct:
            resp.headers["Content-Type"] = ct
        return resp
    except Exception as e:
        return jsonify({"error": str(e)[:160]}), 502


@app.route('/ask', methods=['POST'])
def ask():
    """
    JSON body:
      - message:        (str) son kullanıcı mesajı
      - history:        (list, opsiyonel) [{role:'user'|'assistant', content:'...'}, ...]
      - pdf_context:    (str, opsiyonel) son analiz edilen PDF metni
      - image_context:  (str, opsiyonel) son analiz edilen görselin metinsel özeti
      - dental_context: (dict, opsiyonel) son dental X-ray analiz JSON bağlamı
      - dental_mode:    (bool, opsiyonel) açık dental oturumu

    RAG (yerel data/ indeksi) yalnızca akademik/kaynak sorusu sinyali varsa devreye girer;
    bu modda indeks + DuckDuckGo web özetleri birleştirilebilir; selamlaşmada klasik Groq kullanılır.
    Diş hekimliği soruları dental servise yönlendirilir.
    """
    payload = request.json or {}
    user_message = (payload.get("message") or "").strip()
    history = payload.get("history") or []
    pdf_context = (payload.get("pdf_context") or "").strip()
    image_context = (payload.get("image_context") or "").strip()
    dental_context = payload.get("dental_context")
    dental_mode = bool(payload.get("dental_mode"))

    if not user_message:
        return jsonify({"answer": "Lütfen bir soru yaz."}), 400

    if dental_route_mod.should_route_to_dental(
        user_message,
        dental_context=dental_context if isinstance(dental_context, dict) else None,
        dental_mode=dental_mode,
    ):
        try:
            dental_history = [
                {"role": t.get("role"), "content": t.get("content")}
                for t in history[-12:]
                if t.get("role") in ("user", "assistant") and (t.get("content") or "").strip()
            ]
            answer = dental_mod.chat(
                user_message,
                analysis_context=dental_context if isinstance(dental_context, dict) else None,
                history=dental_history,
            )
            return jsonify({"answer": answer, "handler": "dental"})
        except Exception as e:
            print(f"[/ask] dental yönlendirme hatası: {e}")
            return jsonify({"answer": _dental_unavailable_message(e)}), 503

    if not GROQ_API_KEY:
        return jsonify({
            "answer": "GROQ_API_KEY ortam değişkeni ayarlı değil. .env veya sistem ortamına ekleyip uygulamayı yeniden başlat.",
        }), 503

    try:
        extra_system = _extra_system_from_uploads(pdf_context, image_context)
        if _should_use_indexed_rag(user_message):
            try:
                answer = answer_indexed_with_web(
                    user_message,
                    history=history,
                    extra_system=extra_system,
                )
            except Exception as rag_e:
                print(f"[/ask] indeks/web yolu hata, genel modele düşülüyor: {rag_e}")
                answer = chat_with_model(
                    user_message,
                    extra_system=extra_system,
                    history=history,
                )
        else:
            answer = chat_with_model(
                user_message,
                extra_system=extra_system,
                history=history,
            )
        if not (answer or "").strip():
            answer = "Asistan şu an cevap üretemedi, lütfen tekrar dene."
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"[/ask] Hata: {e}")
        return jsonify({"answer": "Asistan şu an cevap üretemedi, lütfen tekrar dene."}), 500


@app.route("/rag/ingest", methods=["POST"])
def rag_ingest():
    """PDF'leri ./data altından yeniden indeksle. JSON: {\"reset\": true|false}"""
    payload = request.json or {}
    reset = bool(payload.get("reset"))
    try:
        stats = ingest.ingest_all_pdfs(reset=reset)
        return jsonify(stats)
    except Exception as e:
        print(f"[/rag/ingest] Hata: {e}")
        return jsonify({"error": str(e)[:200]}), 500


@app.route("/rag/stats", methods=["GET"])
def rag_stats():
    try:
        col = db_mod.get_collection()
        n = col.count()
        return jsonify({"chunks": n, "chroma_path": db_mod.get_chroma_path()})
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500





@app.route('/analyze-pdf', methods=['POST'])
def analyze_pdf():
    """
    Form alanları:
      - file: PDF dosyası (multipart/form-data)
      - start_page: başlangıç sayfası (1-indexed, opsiyonel — boş ise tümü)
      - end_page: bitiş sayfası (1-indexed, opsiyonel)
      - question: kullanıcının PDF hakkında akademik sorusu
    """
    try:
        if client is None:
            return jsonify({"answer": "GROQ_API_KEY ayarlı değil."}), 503

        pdf_file = request.files.get("file")
        if pdf_file is None:
            return jsonify({"answer": "Herhangi bir PDF dosyası yüklenmedi."}), 400

        start_page_raw = (request.form.get("start_page") or "").strip()
        end_page_raw = (request.form.get("end_page") or "").strip()
        question = (request.form.get("question") or "").strip()

        
        try:
            reader = PdfReader(pdf_file)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    return jsonify({"answer": "Bu PDF şifreli. Lütfen şifresiz bir kopyasını yükle."}), 400
            total_pages = len(reader.pages)
        except Exception as e:
            print(f"[/analyze-pdf] PDF okuma hatası: {e}")
            return jsonify({"answer": f"PDF dosyası açılamadı (bozuk veya desteklenmiyor olabilir). Detay: {str(e)[:140]}"}), 400

        
        
        if not start_page_raw and not end_page_raw:
            start_page = 1
            end_page = min(total_pages, 20)
        else:
            try:
                start_page = max(int(start_page_raw or "1"), 1)
            except ValueError:
                return jsonify({"answer": "Başlangıç sayfası geçerli bir sayı olmalı."}), 400

            if end_page_raw == "":
                end_page = start_page
            else:
                try:
                    end_page = max(int(end_page_raw), start_page)
                except ValueError:
                    return jsonify({"answer": "Bitiş sayfası geçerli bir sayı olmalı."}), 400

        if start_page < 1 or start_page > total_pages:
            return jsonify({"answer": f"Başlangıç sayfası 1 ile {total_pages} arasında olmalı."}), 400
        if end_page < 1 or end_page > total_pages:
            return jsonify({"answer": f"Bitiş sayfası 1 ile {total_pages} arasında olmalı."}), 400

        
        start_idx = start_page - 1
        end_idx = end_page - 1

        extracted_parts = []
        for i in range(start_idx, end_idx + 1):
            try:
                page = reader.pages[i]
                page_text = page.extract_text() or ""
            except Exception as pe:
                print(f"[/analyze-pdf] Sayfa {i+1} okunamadı: {pe}")
                page_text = ""
            extracted_parts.append(f"[Page {i+1}]\n{page_text.strip()}")

        extracted_text = "\n\n".join(extracted_parts).strip()

        if not extracted_text:
            return jsonify({
                "answer": "Seçilen sayfalardan metin çıkarılamadı. PDF taranmış (scan) olabilir veya metin içermiyor olabilir."
            }), 400

        
        MAX_CHARS = 60000
        if len(extracted_text) > MAX_CHARS:
            extracted_text = extracted_text[:MAX_CHARS] + "\n\n[... PDF kısaltıldı: kalan sayfalar gönderilmedi ...]"

        if not question:
            question = (
                "Provide a detailed academic analysis of the selected pages: "
                "main thesis, key arguments, methodology, results, limitations, "
                "and a concise summary. Reply in the user's language; if "
                "unclear, default to Turkish."
            )

        user_content = (
            "The user uploaded a PDF and asked a question about specific pages.\n"
            "TASK: Analyze ONLY the provided page contents and answer the user's "
            "question in their own language (auto-detect; default Turkish).\n\n"
            f"USER QUESTION:\n{question}\n\n"
            f"PDF PAGES (pages {start_page}–{end_page} of {total_pages}):\n"
            f"{extracted_text}"
        )

        
        
        def _do_call(model_name):
            return chat_with_model(user_content, model=model_name)

        answer = None
        last_err = None
        attempts = [
            (HEAVY_MODEL, 0),       
            (HEAVY_MODEL, 4),       
            (TEXT_MODEL,  0),       
        ]
        for model_name, wait_s in attempts:
            try:
                if wait_s:
                    time.sleep(wait_s)
                answer = _do_call(model_name)
                break
            except Exception as ai_err:
                last_err = ai_err
                print(f"[/analyze-pdf] {model_name} hatası: {ai_err}")
                if not _is_rate_limit(ai_err):
                    break

        if answer is None:
            err_msg = str(last_err or "")
            low = err_msg.lower()
            if _is_rate_limit(last_err) if last_err else False:
                friendly = ("Groq'un günlük token kotası dolmuş görünüyor. "
                            "Birkaç dakika bekleyip tekrar dene; sorun devam ederse "
                            "API anahtarının kotası tükenmiş olabilir.")
            elif "context" in low and ("length" in low or "window" in low):
                friendly = "PDF modelin alabileceğinden uzun. Daha küçük bir sayfa aralığı seç (örn. 1-10)."
            elif "401" in err_msg or "api key" in low or "invalid_api_key" in low:
                friendly = "API anahtarında sorun var. Sistem yöneticisine bildir."
            elif "timeout" in low:
                friendly = "İstek zaman aşımına uğradı. Tekrar dene."
            else:
                friendly = f"Asistan PDF'i analiz edemedi. Detay: {err_msg[:160]}"
            return jsonify({"answer": friendly}), 500

        response = {
            "answer": answer,
            "total_pages": total_pages,
            "start_page": start_page,
            "end_page": end_page,
            
            "extracted_text": extracted_text,
        }
        
        _idx_flag = os.environ.get("RAG_INDEX_UPLOADED_PDF_PAGES", "1").strip().lower()
        if _idx_flag not in ("0", "false", "no", "off"):
            try:
                fname = secure_filename(pdf_file.filename or "document.pdf") or "document.pdf"
                src = f"upload:{fname}:p{start_page}-{end_page}"
                side = ingest.upsert_text_as_source(extracted_text, src)
                if side.get("chunks_added"):
                    response["rag_indexed_chunks"] = side["chunks_added"]
            except Exception as ex:
                print(f"[/analyze-pdf] RAG sidecar index skipped: {ex}")

        return jsonify(response)
    except Exception as e:
        print(f"[/analyze-pdf] Beklenmeyen hata: {e}")
        return jsonify({"answer": f"PDF analiz edilirken bir sorun oluştu. Detay: {str(e)[:140]}"}), 500





@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Form alanları:
      - file: görsel dosyası (jpg/png/webp/gif)
      - question: kullanıcının görsel hakkında akademik sorusu (opsiyonel)
    """
    try:
        image_file = request.files.get("file")
        if image_file is None:
            return jsonify({"answer": "Herhangi bir görsel yüklenmedi."}), 400

        question = (request.form.get("question") or "").strip()
        if not question:
            question = (
                "Analyze this image with maximum visual detail using the full "
                "fracture–crack (kırık–çatlak) workflow in your system instructions. "
                "Go bone-by-bone on any X-ray; actively search for BOTH fractures "
                "(displacement, step-off, buckle/torus, complete/incomplete fracture lines) "
                "AND cracks/fissures (hairline lucency, cortical fissure, trabecular crack, "
                "stress lines without displacement). Classify each bone where possible. "
                "Do not call the study normal without per-bone fracture–crack assessment. "
                "End with a bullet list of teachable anatomy/pathology facts from "
                "your findings only. Reply in the user's language; default Turkish."
            )

        
        raw = image_file.read()
        if not raw:
            return jsonify({"answer": "Görsel boş görünüyor."}), 400

        mime = (image_file.mimetype or "").strip().lower()
        if mime and not mime.startswith("image/"):
            return jsonify({"answer": "Lütfen geçerli bir görsel dosyası yükle (JPG, PNG, WEBP, GIF)."}), 400

        data_url, prep_err = vision_image.prepare_vision_image(raw, mime)
        if prep_err:
            return jsonify({"answer": prep_err}), 400

        if client is None:
            return jsonify({"answer": "GROQ_API_KEY ayarlı değil."}), 503

        messages = [
            {
                "role": "system",
                "content": ACADEMIC_SYSTEM_PROMPT + "\n\n" + MEDICAL_IMAGE_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        answer = None
        last_err = None
        for wait_s in (0, 4):
            try:
                if wait_s:
                    time.sleep(wait_s)
                completion = client.chat.completions.create(
                    model=VISION_MODEL,
                    temperature=0.2,
                    max_completion_tokens=2048,
                    reasoning_format="hidden",
                    messages=messages,
                )
                answer = (completion.choices[0].message.content or "").strip()
                break
            except Exception as ai_err:
                last_err = ai_err
                print(f"[/analyze-image] Groq hatası: {ai_err}")
                if not _is_rate_limit(ai_err):
                    break

        if not answer:
            err_msg = str(last_err or "")
            low = err_msg.lower()
            if _is_rate_limit(last_err) if last_err else False:
                friendly = (
                    "Groq kotası dolmuş olabilir. Birkaç dakika bekleyip tekrar dene."
                )
            elif "413" in err_msg or "payload" in low or "too large" in low:
                friendly = "Görsel API limitine takıldı. Daha küçük bir fotoğraf yükle."
            elif "401" in err_msg or "api key" in low or "invalid_api_key" in low:
                friendly = "API anahtarında sorun var. GROQ_API_KEY kontrol edin."
            else:
                friendly = f"Görsel analiz edilemedi. Detay: {err_msg[:160]}"
            return jsonify({"answer": friendly}), 500

        return jsonify({"answer": answer})
    except Exception as e:
        print(f"[/analyze-image] Hata: {e}")
        return jsonify({"answer": f"Görsel analiz edilirken bir sorun oluştu. Detay: {str(e)[:160]}"}), 500





QUIZ_INSTRUCTIONS = """Sen, akademik materyalleri (PDF metinleri ve görsel açıklamaları) analiz ederek öğrencilerin öğrenme süreçlerini ölçen, gerçek bir öğretmenin hazırladığı sınav kalitesinde sorular yazan uzman bir Akademik Sınav Hazırlama Asistanısın.

Görevin, sana sağlanan kaynak metne/veriye %100 sadık kalarak, pedagojik ilkelere uygun, GERÇEK SINAV TİPİNDE 10 soruluk bir çoktan seçmeli test oluşturmaktır. Ürettiğin sorular bir üniversite veya lise sınavında doğrudan kullanılabilecek nitelikte olmalı; öğrenciyi kaynaktaki bir cümleyi ezbere tanımaya değil, konuyu anlayıp anlamadığına göre ölçmelidir.

Soru Kökü Kuralları:
1. Kaynak Uyumluluğu: Tüm sorular ve cevaplar sadece sağlanan materyalde geçen bilgilere dayanmalıdır. Materyalde yer almayan, dışarıdan genel kültür veya varsayımlara dayalı sorular sorma.
2. Kendi Cümlelerinle Sor: Soru kökünü kaynaktaki bir cümleyi birebir kopyalayıp doğru şıkkı da o cümlenin aynısı yaparak KURMA (ör. "X hangi bulgu ile tanımlanır?" diye sorup doğru şıkka kaynaktaki cümleyi aynen yapıştırmak yasaktır). Bunun yerine bilgiyi test edecek gerçek sınav kalıpları kullan: "Neden", "Hangi faktör/kavram/neden", "Aşağıdakilerden hangisi ... değildir", "... ile ... arasındaki fark nedir" gibi anlamayı ölçen sorular kur.
3. Soru Çeşitliliği: 10 sorunun her biri konunun farklı bir boyutunu, kavramını veya analizini ölçmelidir. Üst üste benzer sorular sorma. "Yukarıdaki metne göre..." gibi ifadeleri her soruda tekrarlayarak monotonluk yaratma.

Şık (Cevap) Kuralları — EN KRİTİK KISIM, mutlaka uy:
4. Tek ve Kesin Doğru Cevap: Her sorunun 4 şıkkı (A, B, C, D) olmalı ve bunlardan SADECE VE KESİNLİKLE BİRİ doğru olmalıdır. Doğru cevap tartışmaya açık, kısmen doğru veya yoruma bağlı olmamalı; kaynaktaki bilgiyle birebir, itirazsız doğrulanabilir olmalıdır.
5. Şıklar Kaynaktan Birebir Kopya Cümle OLAMAZ: Doğru şıkkı kaynaktaki bir cümleyi aynen kopyalayarak yazma; bilgiyi kısa ve öz kendi ifadenle yaz.
6. Güçlü ama Açıkça Yanlış Çeldiriciler: Yanlış 3 şık, metindeki BAŞKA bir kavramdan, sayıdan, nedenden veya örnekten türetilmiş, ilk bakışta mantıklı görünen ama kaynakla açıkça çelişen bilgiler olmalıdır.
7. Şıklar Birbirinin Eş Anlamlısı veya Ufak Kelime/Ek Değişikliği OLAMAZ: "yüksek düzeyde / minimal düzeyde", "korunmuştur / korunmamıştır", "artmıştır / azalmıştır" gibi yalnızca tek bir kelimeyi, sıfatı veya olumlu/olumsuz ekini değiştirerek üretilmiş, birbirine neredeyse aynı görünen şıklar üretme. Her şık, kendi başına farklı ve anlamlı bir bilgi/iddia içermelidir; bir öğrenci konuyu hiç anlamasa bile şıklar arasındaki farkı salt kelime oyunuyla çözemesin.
8. Şıklar birbirinin aynısı veya aynı kavramların permütasyonu olamaz.
9. Yanlış şıklar saçma, komik, konu dışı veya dil bilgisi açısından bariz bir şekilde sırıtan olmamalıdır.

Dil ve Ton: Net, akademik, anlaşılır ve kurallara uygun bir Türkçe kullan."""

QUIZ_SYSTEM_PROMPT = (
    QUIZ_INSTRUCTIONS
    + "\n\nYanıtın YALNIZCA geçerli bir JSON nesnesi olsun; markdown, kod bloğu veya ek açıklama yazma."
)

QUIZ_OPTION_LETTERS = ("A", "B", "C", "D")
QUIZ_OPTION_COUNT = 4

QUIZ_META_PHRASES = (
    "radyolog", "klinisyen korelasyon", "klinik korelasyon", "klinisyenle korelasyon",
    "tanı değildir", "tanı koymaz", "kesin tanı", "profesyonel değerlendirme",
    "güvenlik uyarısı", "sorumluluk reddi", "disclaimer", "not a diagnosis",
    "clinical correlation", "radiologist", "clinician correlation",
    "güven düzeyi", "güven seviyesi", "analiz raporu", "asistan raporu",
    "bu bir tanı", "önerilir bu", "önerilir.", "yorum değildir",
)

_ANATOMY_LEXICON = (
    "scaphoid", "skafoid", "lunate", "triquetrum", "pisiform", "trapezium", "trapezoid",
    "capitate", "hamate", "radius", "ulna", "karpal", "carpal", "metacarpal", "metakarpal",
    "femur", "tibia", "fibula", "humerus", "patella", "talus", "calcaneus", "kalkaneus",
    "malleolus", "maléol", "clavicle", "klavikula", "scapula", "skapula", "sternum",
    "vertebra", "omurga", "servikal", "lumbar", "lomber", "sakrum", "pelvis", "kaburga",
    "rib", "mandible", "mandibula", "maxilla", "kafatası", "cranium",
)


def _anatomy_terms_in_text(text: str) -> set[str]:
    blob = (text or "").casefold()
    return {term for term in _ANATOMY_LEXICON if term in blob}


def _quiz_text_has_meta_phrase(text: str) -> bool:
    blob = (text or "").casefold()
    return any(p in blob for p in QUIZ_META_PHRASES)


def _quiz_references_unlisted_anatomy(q: dict, source: str) -> bool:
    allowed = _anatomy_terms_in_text(source)
    blob = (q.get("question") or "") + " " + " ".join(q.get("options") or [])
    mentioned = _anatomy_terms_in_text(blob)
    if not mentioned:
        return False
    if not allowed:
        return True
    return bool(mentioned - allowed)


def _quiz_is_grounded_quality(q: dict, *, source: str, from_image: bool) -> bool:
    qtext_raw = q.get("question") or ""
    if _quiz_text_has_meta_phrase(qtext_raw):
        return False
    qtext = qtext_raw.strip()
    if len(qtext) < 20:
        return False
    if not qtext.endswith("?"):
        return False
    return True

def _normalize_quiz_option_text(text: str) -> str:
    t = str(text or "").strip()
    t = re.sub(r"[\t\r\n]+", " ", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def _coerce_quiz_correct_index(corr, *, option_count: int = QUIZ_OPTION_COUNT) -> int:
    letter_map = {letter: idx for idx, letter in enumerate(QUIZ_OPTION_LETTERS)}
    if isinstance(corr, str):
        s = corr.strip().upper()
        if s in letter_map:
            return letter_map[s]
        if s.isdigit():
            corr = int(s)
    if not isinstance(corr, int):
        return 0
    return max(0, min(option_count - 1, corr))


def _extract_quiz_options(q: dict) -> list[str]:
    raw = (
        q.get("secenekler")
        or q.get("options")
        or q.get("choices")
        or q.get("şıklar")
        or q.get("siklar")
        or []
    )
    if isinstance(raw, dict):
        ordered = []
        for key in QUIZ_OPTION_LETTERS + tuple(l.lower() for l in QUIZ_OPTION_LETTERS):
            val = raw.get(key)
            if val is not None and str(val).strip():
                ordered.append(str(val))
        raw = ordered
    if not isinstance(raw, list):
        return []
    return [_normalize_quiz_option_text(o) for o in raw if str(o).strip()]


def _quiz_option_signature(text: str) -> str:
    """Permütasyon tespiti: aynı kavramların farklı sıralamasını yakala."""
    t = _normalize_quiz_option_text(text).casefold()
    parts = re.split(r"[,;/|•·]+", t)
    tokens = sorted(p.strip() for p in parts if len(p.strip()) > 1)
    if len(tokens) >= 2:
        return "|".join(tokens)
    words = sorted(w for w in re.split(r"\s+", t) if len(w) > 2)
    return " ".join(words) if words else t


def _quiz_options_are_distinct(opts: list[str]) -> bool:
    if len(opts) != QUIZ_OPTION_COUNT:
        return False
    lowered = [o.casefold() for o in opts]
    return len(set(lowered)) == len(opts)


QUIZ_OPTION_SIMILARITY_THRESHOLD = 0.80
QUIZ_OPTION_LEN_RATIO_MIN = 0.80


def _quiz_options_too_similar(opts: list[str]) -> bool:
    """
    Aynı sorudaki 2 şık, tek bir kelime/ek değiştirilerek üretilmiş neredeyse
    aynı cümleyse (ör. "...yüksek düzeydedir" / "...minimal düzeydedir", ya da
    "izlenmektedir" / "izlenmemektedir" gibi bir olumsuzluk eki), permütasyon
    kontrolü bunu yakalamaz çünkü kelime kümesi zaten farklıdır. Burada
    difflib ile karakter bazlı benzerliğe bakılır — ama yalnızca UZUNLUKLARI
    birbirine yakın olan şık çiftleri karşılaştırılır (`len_ratio` eşiği),
    çünkü "kapsamlı doğru cevap / alt küme çeldirici" gibi meşru ve uzunlukça
    çok farklı şık kalıplarını yanlışlıkla elememek gerekir. Modelin talimatı
    görmezden geldiği durumlarda son çare filtresi olarak çalışır.
    """
    texts = [_normalize_quiz_option_text(o).casefold() for o in opts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if not a or not b:
                continue
            len_ratio = min(len(a), len(b)) / max(len(a), len(b))
            if len_ratio < QUIZ_OPTION_LEN_RATIO_MIN:
                continue
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            if ratio >= QUIZ_OPTION_SIMILARITY_THRESHOLD:
                return True
    return False


def _shuffle_quiz_options(opts: list[str], correct: int) -> tuple[list[str], int]:
    indexed = list(enumerate(opts))
    random.shuffle(indexed)
    new_opts = [o for _, o in indexed]
    new_correct = next(i for i, (orig_i, _) in enumerate(indexed) if orig_i == correct)
    return new_opts, new_correct


def _sanitize_quiz_question(
    q: dict,
    *,
    source: str = "",
    from_image: bool = False,
) -> dict | None:
    try:
        qtext = _normalize_quiz_option_text(
            q.get("soru_koku")
            or q.get("question")
            or q.get("soru")
            or q.get("text")
            or ""
        )
        if not qtext:
            return None
        opts = _extract_quiz_options(q)
        if len(opts) < QUIZ_OPTION_COUNT:
            return None
        opts = opts[:QUIZ_OPTION_COUNT]
        if not _quiz_options_are_distinct(opts):
            return None
        if _quiz_options_too_similar(opts):
            return None
        corr = _coerce_quiz_correct_index(
            q.get("dogru_cevap") or q.get("correct") or q.get("dogru") or q.get("answer"),
            option_count=len(opts),
        )
        topic = _normalize_quiz_option_text(q.get("topic") or q.get("konu") or "Genel") or "Genel"
        candidate = {
            "question": qtext,
            "options": opts,
            "correct": corr,
            "topic": topic,
        }
        if not _quiz_is_grounded_quality(candidate, source=source, from_image=from_image):
            return None
        opts, corr = _shuffle_quiz_options(opts, corr)
        candidate["options"] = opts
        candidate["correct"] = corr
        return candidate
    except Exception:
        return None


def _trim_quiz_source(content: str) -> str:
    text = (content or "").strip()
    if len(text) <= QUIZ_SOURCE_MAX_CHARS:
        return text
    return text[:QUIZ_SOURCE_MAX_CHARS] + "\n\n[... kaynak kırpıldı ...]"


def _quiz_json_schema(count: int) -> str:
    return (
        f"Yanıtı yalnızca şu JSON formatında ver (tam {count} soru):\n"
        '{ "questions": [ { "soru_no": 1, "soru_koku": "...", '
        '"secenekler": { "A": "...", "B": "...", "C": "...", "D": "..." }, '
        '"dogru_cevap": "B" }, ... ] }\n'
    )


def _quiz_source_note(*, from_image: bool) -> str:
    if from_image:
        return "Kaynak türü: görsel analiz açıklaması.\n"
    return "Kaynak türü: PDF / akademik metin.\n"


def _build_quiz_user_prompt(
    content: str,
    *,
    from_image: bool,
    count: int = 10,
    batch_no: int = 1,
    batch_total: int = 1,
) -> str:
    scope = ""
    if batch_total > 1:
        if batch_no == 1:
            scope = f"Parça {batch_no}/{batch_total}: testin ilk bölümü; farklı alt konular seç.\n\n"
        else:
            scope = (
                f"Parça {batch_no}/{batch_total}: testin sonraki bölümü; "
                "önceki parçayla aynı soru köklerini tekrarlama.\n\n"
            )
    return (
        scope
        + f"Aşağıdaki kaynak için TAM {count} soruluk çoktan seçmeli test oluştur.\n"
        + _quiz_source_note(from_image=from_image)
        + "\n"
        + _quiz_json_schema(count)
        + "\nKAYNAK:\n"
        + content
    )


def _build_quiz_topup_prompt(
    content: str,
    *,
    from_image: bool,
    existing: list[dict],
    need: int,
) -> str:
    existing_lines = "\n".join(f"- {q['question']}" for q in existing) or "(yok)"
    return (
        f"Aşağıdaki kaynak için TAM {need} YENİ soru daha oluştur.\n"
        "Daha önce kullanılan soruları tekrarlama veya çok yakın parafraz etme.\n"
        + _quiz_source_note(from_image=from_image)
        + "\nKULLANILAN SORULAR:\n"
        + existing_lines
        + "\n\n"
        + _quiz_json_schema(need)
        + "\nKAYNAK:\n"
        + content
    )


def _load_quiz_json(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return {"questions": data}
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                    if isinstance(data, list):
                        return {"questions": data}
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        return {}


def _extract_quiz_question_list(data: dict) -> list:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("questions", "sorular", "quiz", "items", "data"):
        val = data.get(key)
        if isinstance(val, list):
            return val
    return []


def _parse_quiz_questions(raw: str, *, source: str = "", from_image: bool = False) -> list[dict]:
    data = _load_quiz_json(raw)
    questions_in = _extract_quiz_question_list(data)
    out: list[dict] = []
    for q in questions_in:
        if not isinstance(q, dict):
            continue
        cleaned = _sanitize_quiz_question(q, source=source, from_image=from_image)
        if cleaned:
            out.append(cleaned)
    return out


def _quiz_question_key(q: dict) -> str:
    return re.sub(r"\s+", " ", (q.get("question") or "").casefold()).strip()


def _merge_quiz_questions(target: list[dict], incoming: list[dict]) -> None:
    seen = {_quiz_question_key(q) for q in target}
    for q in incoming:
        key = _quiz_question_key(q)
        if not key or key in seen:
            continue
        seen.add(key)
        target.append(q)


def _quiz_token_budget(count: int) -> int:
    """Groq TPM kotası (~6000/istek) için çıktı tokenını sınırla."""
    return min(3000, max(1400, 280 * count + 250))


def _call_quiz_model(user_prompt: str, *, max_tokens: int = 4500) -> str:
    last_err = None
    for attempt in range(2):
        try:
            if attempt:
                time.sleep(_rate_limit_delay(last_err, default=1.5))
            completion = client.chat.completions.create(
                model=QUIZ_MODEL,
                temperature=0.35,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return completion.choices[0].message.content or "{}"
        except Exception as qe:
            last_err = qe
            print(f"[/generate-quiz] {QUIZ_MODEL} hatası (deneme {attempt + 1}): {qe}")
            if not _is_rate_limit(qe):
                break
    raise last_err if last_err else RuntimeError("Quiz üretilemedi.")


def _request_quiz_questions(
    content: str,
    *,
    from_image: bool,
    count: int,
    existing: list[dict] | None = None,
) -> list[dict]:
    if existing:
        prompt = _build_quiz_topup_prompt(
            content, from_image=from_image, existing=existing, need=count,
        )
    else:
        prompt = _build_quiz_user_prompt(content, from_image=from_image, count=count)
    raw = _call_quiz_model(prompt, max_tokens=_quiz_token_budget(count))
    return _parse_quiz_questions(raw, source=content, from_image=from_image)


def _generate_quiz_pool(content: str, *, from_image: bool) -> list[dict]:
    """Sıralı 5+5 (bekleme yok); eksikse tek tamamlama turu."""
    content = _trim_quiz_source(content)
    pool: list[dict] = []

    for part in (1, 2):
        if len(pool) >= QUIZ_TARGET_COUNT:
            break
        try:
            before = len(pool)
            _merge_quiz_questions(
                pool,
                _request_quiz_questions(
                    content, from_image=from_image, count=QUIZ_BATCH_SIZE,
                ),
            )
            print(f"[/generate-quiz] parça {part}/2 pool={len(pool)} (+{len(pool) - before})")
        except Exception as part_err:
            print(f"[/generate-quiz] parça {part} hatası: {part_err}")

    if len(pool) < QUIZ_MIN_RETURN:
        try:
            before = len(pool)
            _merge_quiz_questions(
                pool,
                _request_quiz_questions(
                    content, from_image=from_image, count=QUIZ_BATCH_SIZE,
                ),
            )
            print(f"[/generate-quiz] yedek parça pool={len(pool)} (+{len(pool) - before})")
        except Exception as backup_err:
            print(f"[/generate-quiz] yedek parça hatası: {backup_err}")

    if QUIZ_MIN_RETURN <= len(pool) < QUIZ_TARGET_COUNT:
        need = QUIZ_TARGET_COUNT - len(pool)
        try:
            before = len(pool)
            _merge_quiz_questions(
                pool,
                _request_quiz_questions(
                    content,
                    from_image=from_image,
                    count=need,
                    existing=pool,
                ),
            )
            print(f"[/generate-quiz] tamamlama pool={len(pool)} (+{len(pool) - before})")
        except Exception as topup_err:
            print(f"[/generate-quiz] tamamlama atlandı: {topup_err}")

    return pool[:QUIZ_TARGET_COUNT]


@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    """
    JSON body:
      - pdf_context:   (str, opsiyonel) son analiz edilen PDF metni
      - image_context: (str, opsiyonel) son görsel analizinin metni
    Dönüş:
      - { questions: [ {id, question, options[4], correct(0-3), topic}, ... ] }
    """
    payload = request.json or {}
    pdf_context = (payload.get("pdf_context") or "").strip()
    image_context = (payload.get("image_context") or "").strip()

    content = pdf_context or image_context
    if not content:
        return jsonify({"error": "Quiz oluşturmak için önce bir PDF veya görsel analiz et."}), 400

    from_image = bool(image_context) and not pdf_context

    if client is None:
        return jsonify({"error": "GROQ_API_KEY ayarlı değil."}), 503

    try:
        pool = _generate_quiz_pool(content, from_image=from_image)
        if len(pool) < QUIZ_MIN_RETURN:
            return jsonify({
                "error": "Quiz şu an oluşturulamadı. Lütfen analizi yenileyip tekrar dene.",
            }), 500

        questions = [
            {
                "id": i + 1,
                "question": q["question"],
                "options": q["options"],
                "correct": q["correct"],
                "topic": q["topic"],
            }
            for i, q in enumerate(pool)
        ]
        if len(questions) < QUIZ_TARGET_COUNT:
            print(f"[/generate-quiz] uyarı: {len(questions)}/{QUIZ_TARGET_COUNT} soru döndü")
        return jsonify({
            "questions": questions,
            "target_count": QUIZ_TARGET_COUNT,
            "question_count": len(questions),
        })
    except Exception as e:
        print(f"[/generate-quiz] Hata: {e}")
        msg = str(e)
        if _is_rate_limit(e):
            return jsonify({"error": "Groq kotası şu an dolu. Birkaç dakika bekleyip tekrar dene."}), 429
        return jsonify({"error": f"Quiz oluşturulurken bir hata oluştu. Detay: {msg[:160]}"}), 500


if __name__ == '__main__':
    
    try:
        ingest.ingest_all_pdfs(reset=False)
    except Exception as e:
        print(f"[startup] RAG ingest atlandı: {e}")
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")