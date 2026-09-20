"""
translator.py
--------------
The translation engine.

MODEL LOADING STRATEGY (read this before touching this file)
--------------------------------------------------------------
1. WHEN the model loads: exactly once, the first time `get_translator()`
   is called (in practice: at FastAPI startup, see main.py's lifespan).
2. HOW it is stored: as module-level singleton state inside the
   `Translator` class instance - the tokenizer and model objects live in
   process memory for the lifetime of the running server.
3. HOW inference works: for each request we (a) tell the tokenizer which
   language the input is in via `src_lang`, (b) tokenize the text into
   subword IDs, (c) run `model.generate()` which runs the encoder once
   over the input and then decodes the output token-by-token, forcing
   the very first decoded token to be the target language's ID
   (`forced_bos_token_id`) so the model knows which language to write
   in, and (d) detokenize the generated IDs back into text.
4. WHY load once instead of per-request: loading NLLB-600M means reading
   ~2.4GB of weights off disk into RAM and building the PyTorch compute
   graph - this takes several seconds. Doing that on every API call
   would make every single translation take as long as a cold start,
   which is wasteful and would make the app feel broken. Loading once
   and reusing the same in-memory model for every request is the
   standard pattern for serving any ML model.

This project only ever needs ONE model instance because NLLB is a single
multilingual model covering every direction we need - there is no "one
model per language pair" problem to design around here (see config.py
for why that approach was rejected).
"""

import time
import threading

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MODEL_NAME, MODEL_CACHE_DIR, LANGUAGES, CHUNK_MAX_WORDS
from chunking import chunk_text


class TranslationError(Exception):
    pass


class Translator:
    """Lazily-loaded, thread-safe singleton wrapping the NLLB model."""

    _instance: "Translator | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "Translator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Translator()
        return cls._instance

    def load(self):
        """Idempotent - safe to call multiple times; only loads once."""
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:  # re-check inside the lock
                return
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, cache_dir=MODEL_CACHE_DIR
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME, cache_dir=MODEL_CACHE_DIR
            )
            # CPU-only by design (see config.py docstring). We do not
            # call .to("cuda") - this keeps the project honestly
            # runnable on a normal laptop with no GPU dependency.
            self.model.eval()

    def is_loaded(self) -> bool:
        return self.model is not None

    def _translate_chunk(self, text: str, src_code: str, tgt_code: str) -> str:
        self.tokenizer.src_lang = src_code
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)

        # transformers>=4.35 exposes convert_tokens_to_ids for FLORES codes;
        # older versions expose lang_code_to_id on the tokenizer directly.
        if hasattr(self.tokenizer, "lang_code_to_id"):
            forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_code]
        else:
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_length=256,
                num_beams=4,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def translate(self, text: str, source_language: str, target_language: str):
        """Returns (translated_text, chunks_used, latency_ms)."""
        if not self.is_loaded():
            raise TranslationError("Model is not loaded yet.")

        if source_language not in LANGUAGES:
            raise TranslationError(f"Unsupported source language: {source_language}")
        if target_language not in LANGUAGES:
            raise TranslationError(f"Unsupported target language: {target_language}")

        src_code = LANGUAGES[source_language]
        tgt_code = LANGUAGES[target_language]

        start = time.perf_counter()

        chunks = chunk_text(text, CHUNK_MAX_WORDS)
        if not chunks:
            raise TranslationError("Nothing to translate after preprocessing.")

        try:
            translated_chunks = [
                self._translate_chunk(chunk, src_code, tgt_code) for chunk in chunks
            ]
        except Exception as exc:  # noqa: BLE001 - convert any model error
            raise TranslationError(f"Translation failed: {exc}") from exc

        translated_text = " ".join(translated_chunks)
        latency_ms = (time.perf_counter() - start) * 1000

        return translated_text, len(chunks), latency_ms


def get_translator() -> Translator:
    return Translator.get_instance()
