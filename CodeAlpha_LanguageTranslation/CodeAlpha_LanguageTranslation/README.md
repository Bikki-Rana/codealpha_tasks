# AI-Powered Language Translation Tool
**CodeAlpha AI Internship - Project 3 of 3**

## Overview

A locally-run, offline neural machine translation tool. A user picks a
source and target language, enters text, and gets a translation - no
paid API, no cloud translation service. The heavy lifting is done by a
pretrained multilingual Transformer model (Meta's NLLB-200, distilled)
running entirely on the local machine via Hugging Face Transformers.

Supported languages: **English, Hindi, Bengali, Tamil, Telugu, Marathi,
Gujarati** (any direction between any two of these seven).

## Features

- Multilingual, offline neural translation (no OpenAI/Gemini/Claude/paid API)
- Local AI inference via Hugging Face Transformers + PyTorch (CPU)
- Automatic source-language detection (optional, local, via `langdetect`)
- Manual language selection, always overridable
- One-click language swap (including swapping the text itself)
- Local JSON-based translation history (view / clear)
- Safe long-text handling via sentence-aware chunking
- FastAPI backend with proper validation and error handling
- Clean web UI (plain HTML/CSS/JS, no build step)
- Automated tests (unit + API level)
- Optional BLEU/chrF quality evaluation script
- Local latency measurement script (no invented benchmark numbers)

## Architecture

```
User
 |
Frontend (HTML/CSS/JS)
 |
FastAPI (main.py)
 |
Input Validation (empty / too long / unsupported language / same lang)
 |
Language Detection (optional, langdetect - local, no cloud call)
 |
Chunking (chunking.py - splits long text on sentence boundaries)
 |
Translation Model (translator.py - NLLB-200-distilled-600M, loaded once)
 |
Post-processing (rejoin chunks)
 |
Response (JSON) + History write (history.py)
 |
Frontend renders result
```

## Tech Stack

- Python, FastAPI, Uvicorn
- Hugging Face `transformers`, PyTorch (CPU)
- `sentencepiece` (NLLB tokenizer), `langdetect` (local language ID)
- HTML / CSS / vanilla JavaScript
- `pytest` for tests, `sacrebleu` for optional evaluation

## Why NLLB-200-distilled-600M?

CodeAlpha's task needs 7 languages in both directions (12 total
direction pairs). Helsinki-NLP's OPUS-MT ships one small model **per
direction**, so covering all 12 would mean downloading and holding
~3.5GB across a dozen separate models, with no shared code path and no
easy way to add an 8th language later.

`facebook/nllb-200-distilled-600M` is **one model** that natively
supports all 200 FLORES languages (including all 7 required here) in
every direction. It's a distilled (compressed) version of Meta's larger
NLLB checkpoints - chosen specifically because the full-size NLLB
variants (1.3B/3.3B parameters) are impractical to run on a normal
laptop CPU, while the 600M distilled version can. This keeps the whole
project to one model, one code path, and one honest trade-off: smaller
and faster, at the cost of some fluency compared to bigger models or a
hosted API. That trade-off is stated here, not hidden.

## Installation (Windows)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

First run will download the NLLB model (~2.4GB) into `models/` - this
requires an internet connection once; after that it's cached locally
and every future run is fully offline.

## Running

```bash
cd backend
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000/** in a browser - the backend also
serves the frontend directly, so no separate server is needed. (You can
still open `frontend/index.html` directly if you prefer; update
`API_BASE` in `script.js` to `"http://127.0.0.1:8000"` in that case.)

Interactive API docs: **http://127.0.0.1:8000/docs**

## Example Usage

```
Input Language: English      Target Language: Hindi
Input:  "Where are you going?"
Output: "आप कहाँ जा रहे हैं?"

Input Language: Hindi        Target Language: English
Input:  "मुझे भारत पसंद है।"
Output: "I like India."

Input Language: English      Target Language: Bengali
Input:  "Good morning, how are you?"
Output: "সুপ্রভাত, আপনি কেমন আছেন?"
```

## Measuring latency locally

No latency numbers are invented anywhere in this project. Run this on
your own machine to see real numbers:

```bash
cd backend
python measure_latency.py
```

## Optional quality evaluation (BLEU / chrF)

```bash
pip install sacrebleu
cd backend
python evaluate.py
```

BLEU and chrF are n-gram overlap metrics against reference
translations - useful as a rough sanity check, but they do **not**
measure meaning, fluency, or cultural correctness directly, and they
penalize equally valid rephrasings. No benchmark scores are hardcoded;
`evaluate.py` only has one placeholder example and you're expected to
supply real reference sentences before drawing conclusions.

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests avoid asserting one single "correct" translation string (multiple
valid translations of the same sentence exist); they instead assert
structural correctness - non-empty output, correct language pair
processed, correct HTTP status codes, chunking behaves correctly on
long input, history round-trips correctly, etc.

## Limitations (stated honestly)

- **CPU inference is slow relative to GPU** - expect roughly one to a
  few seconds per sentence on a typical laptop CPU; this is the
  intentional trade-off for zero-GPU-dependency, fully local operation.
- **Model size**: the 600M-parameter distilled model trades some
  fluency for being runnable locally; the full NLLB models (1.3B/3.3B)
  translate better but are not practical without a GPU.
- **Translation ambiguity**: idioms, culturally specific expressions,
  and named entities can be mistranslated or translated too literally.
- **Long-context limitations**: chunk-based translation (see
  `chunking.py`) processes each chunk independently, so cross-paragraph
  coherence (pronoun references, tone) can degrade on very long input.
- **Language coverage**: only the 7 languages explicitly validated here
  are exposed through the API, even though NLLB supports more - adding
  a language means confirming it against the real model, not guessing.
- **Domain-specific terminology** (legal, medical, highly technical
  text) is not specially handled and may translate imprecisely.
- **Language detection** is unreliable on very short text, mixed-language
  text, closely related languages, names, code, or URLs - manual
  selection is always available and recommended when in doubt.

## Future Improvements

- GPU acceleration for lower latency
- Larger/better multilingual models where GPU is available
- Document (Word/PDF) translation, not just plain text
- Voice translation: speech-to-text in, text-to-speech out
- Retrieval-augmented translation for domain-specific terminology
- Docker packaging and cloud deployment
- A persistent evaluation pipeline (BLEU/chrF tracked over time)
- Simple usage analytics for the history data already being collected

## Engineering concepts (quick reference)

1. **Machine translation** - using a model to convert text from one
   language to another automatically.
2. **Transformer** - the neural network architecture (attention-based)
   behind almost all modern MT and language models.
3. **Tokenization** - splitting text into subword units the model
   understands, and mapping them to integer IDs.
4. **Attention** - a mechanism letting the model weigh which input
   tokens matter most when producing each output token.
5. **Encoder-decoder** - the encoder reads and represents the whole
   input; the decoder generates the output one token at a time,
   attending back to the encoder's representation.
6. **Multilingual translation** - one model trained across many
   languages at once, sharing structure between them (e.g. related
   Indic languages help each other) rather than needing N*N models.
7. **Language IDs** - special tokens (e.g. `hin_Deva`) telling the model
   which language the input is in and which to generate.
8. **Inference** - running a trained model forward to get a prediction
   (here: a translation), no training happening.
9. **Why a pretrained model works untrained** - NLLB already learned
   general translation patterns from massive multilingual data; we're
   only reusing that learned knowledge, not teaching it from scratch.
10. **Fine-tuning** - further training a pretrained model on a smaller,
    task/domain-specific dataset to specialize it (not done here - this
    project uses NLLB as-is).
11. **Why quality differs by language** - languages with more training
    data (e.g. Hindi, Bengali) tend to translate better than those NLLB
    saw less of; script and grammatical distance from English matters too.
12. **BLEU / chrF** - automated metrics scoring MT output against
    reference translations via n-gram/character overlap; useful signal,
    not a full substitute for human judgment.
13. **Why GPUs help Transformers** - Transformer inference is dominated
    by large matrix multiplications, which GPUs parallelize far better
    than CPUs, cutting latency substantially.
14. **Scaling to millions of requests** - batch multiple requests
    together, run multiple model replicas behind a load balancer, use
    GPU inference servers, and cache repeated translations.
15. **When to use an API instead of self-hosting** - when you need
    higher quality than a laptop-sized model can give, don't want to
    manage infrastructure, or need language coverage beyond what you can
    self-host - at the cost of per-request pricing and a network
    dependency.

## Project Structure

```
CodeAlpha_LanguageTranslation/
├── backend/
│   ├── main.py              # FastAPI app, routes, validation
│   ├── translator.py        # Model loading + inference
│   ├── language_detector.py # Optional local language detection
│   ├── chunking.py          # Long-text splitting/rejoining
│   ├── history.py           # JSON-based translation history
│   ├── config.py            # Model choice, languages, limits
│   ├── measure_latency.py   # Local latency measurement
│   └── evaluate.py          # Optional BLEU/chrF evaluation
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── models/                  # HF model cache (downloaded on first run)
├── history/
│   └── translations.json
├── tests/
│   ├── test_translation.py
│   ├── test_api.py
│   └── test_validation.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```
