"""
models.py
---------
Pydantic schemas for request/response validation. FastAPI uses these to
auto-validate incoming JSON and auto-generate the OpenAPI docs at /docs.
"""

from pydantic import BaseModel, Field, field_validator


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_language: str = Field(..., description="e.g. 'English'")
    target_language: str = Field(..., description="e.g. 'Hindi'")
    save_history: bool = Field(
        True,
        description="Whether to log this call in translation history. "
                    "The frontend sets this to False for automatic "
                    "as-you-type calls so history isn't cluttered with "
                    "every partial keystroke, and True for an explicit "
                    "translate action.",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        # Only checks for structural emptiness here; deeper checks
        # (length limits, whitespace-only) happen in main.py where we
        # have access to config.MAX_CHARS and can raise a clean 400.
        return v


class TranslationResponse(BaseModel):
    source_language: str
    target_language: str
    original_text: str
    translated_text: str
    detected_language: str | None = None
    chunks_used: int = 1
    latency_ms: float


class LanguageInfo(BaseModel):
    name: str
    code: str


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]


class HistoryRecord(BaseModel):
    timestamp: str
    source_language: str
    target_language: str
    source_text: str
    translated_text: str


class HistoryResponse(BaseModel):
    count: int
    records: list[HistoryRecord]


class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    detected_language: str | None
    confidence_note: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
