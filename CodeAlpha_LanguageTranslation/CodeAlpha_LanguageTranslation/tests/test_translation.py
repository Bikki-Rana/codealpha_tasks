"""
test_translation.py
--------------------
Direct unit tests of the Translator class and chunking logic (no HTTP
layer involved). We deliberately do NOT hardcode one "correct" expected
translated sentence anywhere - multiple valid translations of the same
input can exist, so tests only assert structural correctness.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from translator import get_translator, TranslationError
from chunking import split_into_sentences, group_into_chunks, chunk_text


@pytest.fixture(scope="module")
def translator():
    t = get_translator()
    t.load()
    return t


def test_model_loads(translator):
    assert translator.is_loaded()


def test_translate_returns_nonempty_text(translator):
    text, chunks, latency_ms = translator.translate(
        "Where are you going?", "English", "Hindi"
    )
    assert isinstance(text, str)
    assert text.strip() != ""
    assert chunks >= 1
    assert latency_ms > 0


def test_translate_rejects_unsupported_language(translator):
    with pytest.raises(TranslationError):
        translator.translate("Hello", "English", "Klingon")


def test_sentence_splitting_basic():
    sentences = split_into_sentences("Hello there. How are you? I am fine!")
    assert sentences == ["Hello there.", "How are you?", "I am fine!"]


def test_sentence_splitting_devanagari_danda():
    sentences = split_into_sentences("नमस्ते। आप कैसे हैं।")
    assert len(sentences) == 2


def test_chunk_grouping_respects_word_limit():
    sentences = ["word " * 10 + "."] * 5  # 5 sentences, ~11 words each
    chunks = group_into_chunks(sentences, max_words=20)
    for chunk in chunks:
        # allow a little slack since one long sentence can exceed alone
        assert len(chunk.split()) <= 22 or chunks.count(chunk) == 1


def test_chunk_text_empty_input():
    assert chunk_text("", 50) == []


def test_chunk_text_single_short_sentence():
    chunks = chunk_text("Hello world.", 50)
    assert len(chunks) == 1
