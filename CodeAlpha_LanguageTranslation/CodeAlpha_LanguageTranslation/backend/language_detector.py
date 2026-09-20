"""
language_detector.py
---------------------
Optional automatic source-language detection using `langdetect`
(a local, offline port of Google's language-detection library - no
cloud API call, no network dependency at inference time).

The user can always override the detected language manually; detection
is a convenience default, never a hard requirement.

Honest limitations (explained to the user in the API response too):
- Very short text (a few words) gives langdetect very little signal,
  so confidence is low and results can flip between similar languages.
- Mixed-language / code-switched text (common in casual Indian-language
  typing) confuses statistical detectors, since they assume one
  dominant language per document.
- Similar languages (e.g. Hindi vs Marathi, both Devanagari script)
  are genuinely hard to tell apart from a short snippet.
- Proper names, code snippets, and URLs are not natural-language text
  and can be misclassified as whatever language they happen to
  statistically resemble.
"""

from langdetect import detect, DetectorFactory, LangDetectException

from config import LANGDETECT_TO_NAME

# Make detection deterministic (langdetect is otherwise seeded randomly,
# which can give different answers on the same input across runs).
DetectorFactory.seed = 0

CONFIDENCE_NOTE = (
    "Automatic detection can be unreliable for very short text, "
    "mixed-language text, closely related languages, names, code, "
    "or URLs. Please verify or set the source language manually."
)


def detect_language(text: str) -> str | None:
    """Returns one of our supported language names, or None if the
    detected language isn't one we support / detection failed."""
    text = text.strip()
    if not text:
        return None
    try:
        iso_code = detect(text)
    except LangDetectException:
        return None
    return LANGDETECT_TO_NAME.get(iso_code)
