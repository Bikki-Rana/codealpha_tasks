"""
main.py
-------
FastAPI application: routes, validation, error handling, and model
lifecycle management.

The model is loaded ONCE at process startup (see the `lifespan` context
manager below), not on every request - see translator.py's docstring for
the full reasoning.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import LANGUAGES, MAX_CHARS
from models import (
    TranslationRequest, TranslationResponse,
    LanguagesResponse, LanguageInfo,
    HistoryResponse, HistoryRecord,
    DetectRequest, DetectResponse,
)
from translator import get_translator, TranslationError
from language_detector import detect_language, CONFIDENCE_NOTE
import history as history_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("translation_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading translation model (this happens once)...")
    translator = get_translator()
    translator.load()
    logger.info("Model loaded. Ready to serve requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="CodeAlpha AI-Powered Language Translation Tool",
    description="Local, offline multilingual translation (NLLB-200-distilled-600M) "
                "for English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the local frontend (served from a different port, or opened as a
# file://) to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    translator = get_translator()
    return {
        "status": "ok" if translator.is_loaded() else "model_loading",
        "model_loaded": translator.is_loaded(),
    }


# ---------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------
@app.get("/languages", response_model=LanguagesResponse)
def get_languages():
    return LanguagesResponse(
        languages=[LanguageInfo(name=name, code=code) for name, code in LANGUAGES.items()]
    )


# ---------------------------------------------------------------------
# Translate
# ---------------------------------------------------------------------
@app.post("/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
    text = req.text.strip()

    # --- validation -----------------------------------------------------
    if not text:
        raise HTTPException(status_code=400, detail="[ERROR] Empty input.")

    if len(text) > MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"[ERROR] Text is too long (max {MAX_CHARS} characters).",
        )

    if req.source_language not in LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"[ERROR] Unsupported language: {req.source_language}",
        )
    if req.target_language not in LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"[ERROR] Unsupported language: {req.target_language}",
        )
    if req.source_language == req.target_language:
        raise HTTPException(
            status_code=400,
            detail="[ERROR] Source and target language must differ.",
        )

    translator = get_translator()
    if not translator.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="[ERROR] Translation model unavailable. Still loading, try again shortly.",
        )

    try:
        translated_text, chunks_used, latency_ms = translator.translate(
            text, req.source_language, req.target_language
        )
    except TranslationError as exc:
        logger.error("Translation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"[ERROR] Translation failed.") from exc

    if req.save_history:
        history_store.add_record(
            req.source_language, req.target_language, text, translated_text
        )

    return TranslationResponse(
        source_language=req.source_language,
        target_language=req.target_language,
        original_text=text,
        translated_text=translated_text,
        chunks_used=chunks_used,
        latency_ms=round(latency_ms, 1),
    )


# ---------------------------------------------------------------------
# Optional: language detection
# ---------------------------------------------------------------------
@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="[ERROR] Empty input.")
    detected = detect_language(req.text)
    return DetectResponse(detected_language=detected, confidence_note=CONFIDENCE_NOTE)


# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------
@app.get("/history", response_model=HistoryResponse)
def get_history():
    records = history_store.get_history()
    return HistoryResponse(
        count=len(records),
        records=[HistoryRecord(**r) for r in records],
    )


@app.delete("/history")
def delete_history():
    history_store.clear_history()
    return {"status": "cleared"}


# ---------------------------------------------------------------------
# Serve the frontend (optional convenience: opens on the same port)
# ---------------------------------------------------------------------
import os
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
