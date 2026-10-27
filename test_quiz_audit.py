"""Quiz üretimini test eder: şık tekrarı, permütasyon, öğretmen testi kalitesi."""
from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app as app_mod

SAMPLE_PDF = """
İşletmenin Genel Çevresi ve Stratejik Yönetim — Özet

İşletme, faaliyetlerini sürdürebilmek için dış çevre ile sürekli etkileşim halindedir.
Genel çevre; ekonomik, teknolojik, sosyo-kültürel, politik-yasal ve demografik boyutlardan oluşur.

Ekonomik çevre: enflasyon, faiz oranları, işsizlik ve döviz kurları işletmenin maliyet ve talep koşullarını etkiler.
Teknolojik çevre: yenilikler, otomasyon ve Ar-Ge yatırımları rekabet avantajı yaratır.
Sosyo-kültürel çevre: tüketici yaşam tarzı, değerler ve demografik eğilimler ürün tercihlerini şekillendirir.
Politik-yasal çevre: vergi düzenlemeleri, iş kanunu ve sektörel denetimler işletme kararlarını sınırlar.

SWOT analizi; güçlü yönler, zahmetli yönler, fırsatlar ve tehditlerin sistematik değerlendirilmesidir.
PEST analizi ise politik, ekonomik, sosyo-kültürel ve teknolojik faktörlerin incelenmesine odaklanır.

Rekabet stratejileri: maliyet liderliği, farklılaştırma ve odaklanma (niş pazar) olarak sınıflandırılır.
Porter'un beş kuvvet modeli; sektördeki rekabet yoğunluğu, yeni girenler, ikame ürünler, tedarikçi ve alıcı gücünü analiz eder.
"""

SAMPLE_IMAGE = """
Görsel Analiz Özeti — El bileği AP radyografisi

Modality: X-ray, AP projeksiyon, sol el bileği ve distal önkol.

Bulgular:
- Distal radius kortikal hattında yer değiştirmemiş transvers kırık hattı izlenmektedir.
- Ulnar styloid bölgesinde kortikal süreklilik korunmuştur.
- Karpal kemiklerde belirgin step-off veya yer değiştirme saptanmamıştır.
- Yumuşak doku şişliği minimal düzeydedir.

Yorum: Distal radius yer değiştirmemiş kırık ön planda; immobilizasyon ve ortopedi konsültasyonu önerilir.
Bu rapor kesin klinik tanı değildir; radyolog ve klinisyen korelasyonu gerekir.
"""


def audit_question(q: dict, qnum: int) -> list[str]:
    issues: list[str] = []
    opts = q.get("options") or []
    letters = "ABCD"

    if len(opts) != 4:
        issues.append(f"Soru {qnum}: {len(opts)} şık (4 olmalı)")

    lowered = [o.casefold() for o in opts]
    for i, a in enumerate(lowered):
        for j, b in enumerate(lowered):
            if j > i and a == b:
                issues.append(f"Soru {qnum}: TAM AYNI şık — {letters[i]} ve {letters[j]}: {opts[i]!r}")

    sigs = [app_mod._quiz_option_signature(o) for o in opts]
    for i, sa in enumerate(sigs):
        for j, sb in enumerate(sigs):
            if j > i and sa == sb and opts[i].casefold() != opts[j].casefold():
                issues.append(
                    f"Soru {qnum}: PERMÜTASYON benzeri — {letters[i]} vs {letters[j]}: "
                    f"{opts[i]!r} / {opts[j]!r}"
                )

    qtext = (q.get("question") or "").casefold()
    meta = ("radyolog", "klinisyen korelasyon", "tanı değildir", "yukarıdaki metne göre")
    for m in meta:
        if m in qtext:
            issues.append(f"Soru {qnum}: Meta/uyarı ifadesi soruda: {m!r}")

    if len(qtext) < 20:
        issues.append(f"Soru {qnum}: Soru kökü çok kısa")
    if not qtext.strip().endswith("?"):
        issues.append(f"Soru {qnum}: Soru kökü soru işareti ile bitmiyor (öğretmen formatı zayıf)")

    return issues


def print_quiz(pool: list[dict], title: str) -> list[str]:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Üretilen soru sayısı: {len(pool)}/10\n")

    all_issues: list[str] = []
    for i, q in enumerate(pool, 1):
        letters = "ABCD"
        print(f"--- Soru {i} [{q.get('topic', 'Genel')}] ---")
        print(q["question"])
        for j, opt in enumerate(q["options"]):
            mark = " [DOGRU]" if j == q["correct"] else ""
            print(f"  {letters[j]}) {opt}{mark}")
        if q.get("explanation"):
            print(f"  Açıklama: {q['explanation'][:120]}...")
        print()
        all_issues.extend(audit_question(q, i))

    return all_issues


def run_one(content: str, from_image: bool, label: str) -> list[str]:
    print(f"\n>>> API çağrısı: {label} ...")
    try:
        pool = app_mod._generate_quiz_pool(content, from_image=from_image)
    except Exception as e:
        print(f"HATA: {e}")
        return [f"{label}: API hatası — {e}"]
    return print_quiz(pool, label)


def main() -> int:
    if not app_mod.client:
        print("GROQ_API_KEY yok — test yapılamıyor.")
        return 1

    issues_pdf = run_one(SAMPLE_PDF, False, "PDF — İşletme / Genel Çevre")
    issues_img = run_one(SAMPLE_IMAGE, True, "Görsel — El bileği radyografisi")

    all_issues = issues_pdf + issues_img
    print("\n" + "=" * 72)
    print("DENETİM ÖZETİ")
    print("=" * 72)
    if not all_issues:
        print("Şık tekrarı / permütasyon / belirgin format sorunu bulunamadı.")
        print("Sorular genel olarak öğretmen testi formatına yakın görünüyor.")
        return 0

    print(f"Toplam {len(all_issues)} uyarı/sorun:\n")
    for item in all_issues:
        print(f"  - {item}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
