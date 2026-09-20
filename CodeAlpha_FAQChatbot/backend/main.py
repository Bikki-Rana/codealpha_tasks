import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from . import config
    from .chatbot import FAQChatbot
except ImportError:
    import config
    from chatbot import FAQChatbot

FRONTEND_DIR = os.path.join(config.BASE_DIR, "frontend")

app = FastAPI(title="CodeAlpha FAQ Chatbot API", version="1.0.0")

# Allows the frontend to be opened separately (e.g. a different port) during
# development. When served from this same app (recommended - see README),
# requests are same-origin anyway and this has no effect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot: Optional[FAQChatbot] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    category: Optional[str]
    confidence: float
    matched_question: Optional[str]
    session_id: str


class ClearRequest(BaseModel):
    session_id: str


@app.on_event("startup")
def load_chatbot():
    global chatbot
    print("[INFO] Starting up FAQ Chatbot API...")
    chatbot = FAQChatbot()  # raises clearly on missing data/model - see embeddings.py
    print("[INFO] Chatbot ready. API is live.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": chatbot is not None}


@app.get("/faqs")
def get_faqs():
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized yet.")
    # Only expose question/category - never internal ids or raw embeddings.
    return [{"question": f["question"], "category": f["category"]} for f in chatbot.engine.faqs]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized yet.")

    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = chatbot.ask(req.message, session_id=session_id)
    except Exception as e:
        print(f"[ERROR] /chat failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error while processing your question.")

    return ChatResponse(
        answer=result["answer"],
        category=result["category"],
        confidence=result["confidence"],
        matched_question=result["matched_question"],
        session_id=session_id,
    )


@app.post("/clear")
def clear(req: ClearRequest):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized yet.")
    chatbot.clear_history(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


# --- Serve the frontend from the same server (Phase 5) ---
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if not os.path.exists(index_path):
            raise HTTPException(status_code=404, detail="Frontend not found.")
        return FileResponse(index_path)
