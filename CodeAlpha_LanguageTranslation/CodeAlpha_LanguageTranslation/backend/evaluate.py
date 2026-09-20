"""
evaluate.py
-----------
OPTIONAL translation-quality evaluation using BLEU and chrF (via
sacrebleu). Requires: pip install sacrebleu

BLEU and chrF compare a machine translation against one or more
human "reference" translations using n-gram / character-gram overlap.
They give a single number (roughly 0-100) that correlates loosely with
human judgment - but they do NOT measure meaning, fluency, or cultural
appropriateness directly, and they penalize equally-valid rephrasings
that just happen to use different words than the reference. Treat these
scores as a rough sanity check, never as ground truth about quality.

No benchmark numbers are hardcoded anywhere in this project - run this
yourself against your own reference sentences to get real numbers.

Usage:
    cd backend
    python evaluate.py
"""

from translator import get_translator

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

# Fill in with real reference translations you trust before relying on
# this - these are placeholders illustrating the mechanism, not verified
# gold references.
EVAL_SET = [
    {
        "source_language": "English",
        "target_language": "Hindi",
        "text": "Where are you going?",
        "reference": "आप कहाँ जा रहे हैं?",
    },
]

if __name__ == "__main__":
    if sacrebleu is None:
        print("sacrebleu is not installed. Run: pip install sacrebleu")
        raise SystemExit(1)

    translator = get_translator()
    translator.load()

    hypotheses, references = [], []
    for item in EVAL_SET:
        translated, _, _ = translator.translate(
            item["text"], item["source_language"], item["target_language"]
        )
        print(f"MT:  {translated}")
        print(f"REF: {item['reference']}")
        hypotheses.append(translated)
        references.append(item["reference"])

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    print(f"\nBLEU: {bleu.score:.2f}")
    print(f"chrF: {chrf.score:.2f}")
    print("\nReminder: these scores depend entirely on the reference set")
    print("above being high-quality; expand EVAL_SET with real data before")
    print("drawing conclusions.")
