"""
chunking.py
-----------
Long-text handling.

Transformer translation models have a fixed maximum input length (NLLB's
tokenizer truncates well before ~512 tokens). If we just truncate long
text we silently lose content, so instead we:

    Long Input -> split into sentences -> group sentences into chunks
    that stay under CHUNK_MAX_WORDS -> translate each chunk -> rejoin

Splitting on sentence boundaries (., !, ?, and Indic danda '।') instead of
a hard word cut means we never break a sentence mid-way through, which
would confuse the model and produce garbled output.

Known limitation (stated honestly): translating chunk-by-chunk means the
model has no visibility into a chunk's surrounding context, so pronoun
references, tone, and discourse-level coherence across paragraph
boundaries can degrade compared to translating the whole document at
once. This is an inherent trade-off of chunk-based translation, not a
bug - it's the same trade-off every practical MT system with a bounded
context window makes.
"""

import re

# Splits after ., !, ?, or the Devanagari/Indic full stop '।', while
# keeping the punctuation attached to the sentence it ends.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def group_into_chunks(sentences: list[str], max_words: int) -> list[str]:
    """Greedily pack sentences into chunks <= max_words words each.
    A single sentence longer than max_words becomes its own chunk rather
    than being cut mid-sentence."""
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        n_words = len(sentence.split())
        if current and current_words + n_words > max_words:
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = n_words
        else:
            current.append(sentence)
            current_words += n_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_text(text: str, max_words: int) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []
    return group_into_chunks(sentences, max_words)
