from __future__ import annotations

import shutil
import uuid
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from llm_service import generate_chat_response
from matching import match_findings_to_teeth
from model_service import model_service, strip_internal_masks
from visualization import create_annotated_image


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
DISCLAIMER = (
    "This AI result is for educational and preliminary analysis purposes only. "
    "It does not replace professional dental diagnosis."
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DentAI Vision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.3.154:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5180",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    analysis_context: dict[str, Any] | None = None
    history: list[ChatMessage] = Field(default_factory=list)


@app.on_event("startup")
def startup_event() -> None:
    model_service.load()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "models_ready": model_service.ready,
        "model_error": model_service.load_error,
        "disclaimer": DISCLAIMER,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    if file is None:
        raise HTTPException(status_code=400, detail="No image uploaded.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a JPEG, PNG, WEBP, BMP, or TIFF image.",
        )

    if not model_service.ready:
        return _error_response(model_service.load_error or "Models are not ready.", status_code=503)

    suffix = _safe_suffix(file.filename)
    request_id = uuid.uuid4().hex
    original_path = OUTPUT_DIR / f"{request_id}_original{suffix}"
    annotated_path = OUTPUT_DIR / f"{request_id}_annotated.jpg"

    try:
        with original_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        analysis = model_service.analyze(original_path)
        teeth = analysis["teeth"]
        findings = analysis["findings"]
        image_shape = analysis["image_shape"]
        anatomy_summary = analysis.get("anatomy_summary", {})
        removed_duplicates = analysis.get("removed_duplicates", [])

        matched_results = match_findings_to_teeth(teeth, findings, image_shape)
        create_annotated_image(original_path, teeth, findings, annotated_path)
        logger.info("Pathology matching complete: findings=%s matched_results=%s", len(findings), len(matched_results))

        warnings = []
        if not teeth:
            warnings.append("No teeth were detected by the tooth numbering model.")
        if not findings:
            warnings.append("No dental findings were detected by the finding segmentation model.")
        
        public_teeth = strip_internal_masks(teeth)
        public_findings = strip_internal_masks(findings)
        return JSONResponse(
            {
                "success": True,
                "image_id": request_id,
                "original_image": f"/outputs/{original_path.name}",
                "annotated_image": f"/outputs/{annotated_path.name}",
                "annotated_image_url": f"/outputs/{annotated_path.name}",
                "teeth": public_teeth,
                "tooth_predictions": public_teeth,
                "findings": public_findings,
                "matched_results": matched_results,
                "matches": matched_results,
                "ai_explanation": None,
                "removed_duplicates": removed_duplicates,
                "anatomy_summary": anatomy_summary,
                "warnings": warnings,
                "disclaimer": DISCLAIMER,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        return _error_response(str(exc), status_code=500)
    finally:
        await file.close()


@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer = generate_chat_response(
        question=question,
        analysis_context=request.analysis_context,
        history=[_model_to_dict(message) for message in request.history],
    )

    return JSONResponse(
        {
            "success": True,
            "answer": answer,
            "disclaimer": DISCLAIMER,
        }
    )


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        return suffix
    return ".png"


def _error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": message,
            "teeth": [],
            "tooth_predictions": [],
            "findings": [],
            "matched_results": [],
            "matches": [],
            "ai_explanation": None,
            "removed_duplicates": [],
            "anatomy_summary": {},
            "warnings": [message],
            "disclaimer": DISCLAIMER,
        },
    )
