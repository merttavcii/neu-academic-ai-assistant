"""Resize/compress images for Groq vision (base64 request max ~4 MB)."""

from __future__ import annotations

import base64
from io import BytesIO

# Groq: base64-encoded image body must stay under 4 MB
MAX_B64_BYTES = 3_500_000
MAX_DIM = 1280


def prepare_vision_image(raw: bytes, mime: str = "") -> tuple[str, str | None]:
    """
    Return (data_url, error_message).
    error_message is None on success.
    """
    if not raw:
        return "", "Görsel boş görünüyor."

    try:
        from PIL import Image
    except ImportError:
        est_b64 = (len(raw) * 4 + 2) // 3
        if est_b64 > MAX_B64_BYTES:
            return "", (
                "Görsel çok büyük (Groq en fazla ~4 MB base64 kabul eder). "
                "Kurulum: pip install Pillow"
            )
        mt = mime if mime.startswith("image/") else "image/jpeg"
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mt};base64,{b64}", None

    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception:
        return "", "Görsel dosyası okunamadı. JPG, PNG veya WEBP deneyin."

    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)

    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    for _ in range(8):
        for quality in (85, 75, 65, 55, 45, 35):
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            jpeg = buf.getvalue()
            if (len(jpeg) * 4 + 2) // 3 <= MAX_B64_BYTES:
                b64 = base64.b64encode(jpeg).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}", None
        nw = max(1, int(img.width * 0.75))
        nh = max(1, int(img.height * 0.75))
        if (nw, nh) == img.size:
            break
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    return "", "Görsel sıkıştırıldıktan sonra bile çok büyük. Daha küçük bir fotoğraf yükleyin."
