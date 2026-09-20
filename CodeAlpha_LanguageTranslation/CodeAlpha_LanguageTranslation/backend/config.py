"""
config.py
---------
Central configuration for the translation service.

Why NLLB-200-distilled-600M instead of one Helsinki-NLP OPUS-MT model per
language pair?

- CodeAlpha's task needs EN<->HI/BN/TA/TE/MR/GU, i.e. 12 directions.
  OPUS-MT ships one small model PER direction (~300MB each), so covering
  all 12 would mean downloading and keeping ~3.5GB of separate models in
  memory, and adding a new language later means finding/loading yet
  another model.
- facebook/nllb-200-distilled-600M is ONE model that natively supports
  200 languages (including all 7 required here) in every direction.
  It is a distilled (compressed) version of Meta's much larger NLLB
  model, sized specifically so it can run inference on a normal CPU in
  a few seconds per sentence - the full 1.3B/3.3B NLLB variants are not
  practical on a laptop CPU and are intentionally NOT used here.
- One model loaded once = simpler memory management, one code path for
  every language pair, and trivial to extend (just add the language's
  NLLB code to LANGUAGES below - no new model download).

Trade-off (stated honestly, not hidden): NLLB-600M is smaller/faster
but less fluent than the full-size NLLB models or a hosted API. That is
the correct trade-off for "runs on a normal Windows laptop, CPU-only".
"""

import os

# --- Model ---------------------------------------------------------------
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Where Hugging Face should cache downloaded model weights.
# Keeping this inside the project (models/) makes the project self
# contained and easy to .gitignore.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# --- Supported languages --------------------------------------------------
# NLLB identifies languages with FLORES-200 codes, e.g. "eng_Latn".
# These 7 are the ones this project actually validated against the model's
# real vocabulary - do NOT add a language here unless you have confirmed
# NLLB-200 supports it, since the API will trust this list completely.
LANGUAGES: dict[str, str] = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
}

# Maps langdetect's ISO 639-1 output to the language names above, for the
# optional auto-detect feature.
LANGDETECT_TO_NAME: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
}

# --- Text limits -----------------------------------------------------------
MAX_CHARS = 5000          # reject requests larger than this outright
CHUNK_MAX_WORDS = 60      # ~ words per chunk before we split on sentence
                          # boundaries (rough proxy for the model's token
                          # limit; keeps chunks safely under 512 tokens)

# --- History ---------------------------------------------------------------
HISTORY_DIR = os.path.join(BASE_DIR, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "translations.json")
HISTORY_MAX_RECORDS = 200  # keep the file bounded
os.makedirs(HISTORY_DIR, exist_ok=True)
