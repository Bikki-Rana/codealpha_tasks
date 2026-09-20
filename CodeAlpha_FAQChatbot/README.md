# CodeAlpha FAQ Chatbot - Jharkhand Rai University (JRU)

An intelligent FAQ chatbot for **Jharkhand Rai University (JRU), Ranchi** (jru.edu.in)
that answers natural-language questions by semantic search over a curated FAQ
knowledge base - not keyword matching, and not a hardcoded if/else tree.

## Overview

Students ask questions in plain English ("Which papers do I need to join?") and the
system finds the closest matching FAQ ("What documents are required for admission?")
by comparing sentence embeddings, even when the wording is completely different. If
no FAQ is a close enough match, the bot says so honestly instead of guessing.

**A note on data accuracy:** the FAQ content in `data/faqs.json` is based on publicly
available JRU information as of this writing. Anything that changes by academic
session (fees, exact deadlines, hostel charges, exam dates) is intentionally phrased
to point students to the official portal (jru.edu.in) or the relevant office/phone
number, rather than a hard-coded figure that could go stale. Always verify
session-specific numbers against the official site before relying on this chatbot
for anything decision-critical.

## Features

- Natural-language question answering (no exact-phrase matching required)
- Semantic matching via sentence embeddings + cosine similarity
- Configurable confidence threshold with an explicit "I don't know" fallback
- Per-session conversation history (see Limitations)
- Category detection for each matched answer
- Clean, responsive chat UI with suggested questions and a clear-conversation button
- Runs fully locally on CPU - no paid LLM APIs involved

## Architecture

```
User
 |
Web Interface (HTML/CSS/JS)
 |
FastAPI (backend/main.py)
 |
Text Embedding (backend/embeddings.py - sentence-transformers)
 |
Semantic Similarity (cosine similarity, scikit-learn)
 |
FAQ Retrieval (data/faqs.json)
 |
Confidence Check (backend/chatbot.py)
 |
Response
```

## Tech Stack

Python, FastAPI, Sentence Transformers (`all-MiniLM-L6-v2`), NumPy, scikit-learn,
HTML/CSS/JavaScript, pytest.

## Project Structure

```
CodeAlpha_FAQChatbot/
├── backend/
│   ├── main.py        # FastAPI app, routes, serves the frontend
│   ├── chatbot.py      # threshold gating, unknown-question handling, history
│   ├── embeddings.py    # FAQSemanticSearch: encoding + cosine similarity
│   └── config.py       # model name, paths, confidence threshold
├── data/
│   └── faqs.json        # 45 FAQ entries across 12 categories (JRU-specific)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/
│   └── test_chatbot.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation (Windows)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face -
this requires an internet connection once; afterward it's cached locally.

## Running

```bash
uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser - the FastAPI server serves the
chat UI directly, so there's nothing separate to start for the frontend.

## Example Questions

- "What documents are required for admission?"
- "Which papers are required to join?"
- "What is the eligibility for admission to B.Tech CSE?"
- "Can I get admission through lateral entry?"
- "What courses are offered by the university?"
- "When are the semester exams conducted?"
- "What is the minimum attendance required?"
- "What are the library working hours?"
- "Is the hostel booking amount refundable?"
- "Which companies visit the campus for placements?"
- "How can I apply for a merit scholarship?"
- "How can I contact the admissions office?"

## Testing

```bash
pytest tests/test_chatbot.py -v
```

Covers: exact-match questions, paraphrased questions, unrelated/out-of-scope
questions, empty input, low-confidence fallback, the `/health` check, `/faqs`
listing, and session-clearing.

## Limitations

- This is a **retrieval-based** system, not a generative LLM - it can only return
  answers that already exist in `data/faqs.json`, verbatim.
- Conversation history is stored per session for display purposes, but it is
  **not** used to resolve context in follow-up questions (e.g. asking "is food
  included?" right after a hostel-fees question will not automatically connect
  the two) - that would require query rewriting or an LLM, which this project
  intentionally avoids.
- Answers are only as good as the FAQ dataset - a question about something
  genuinely outside these categories will correctly get the fallback message.
- Session-dependent facts (fees, hostel charges, exact deadlines, exam dates,
  transport routes) are deliberately answered by pointing to the official
  portal or office rather than a fixed number, since JRU's own site indicates
  these change by academic year (the site is already showing the 2026-27
  cycle at the time this was written).
- The confidence threshold (`0.55` by default, in `backend/config.py`) is a
  reasonable starting point for MiniLM but not benchmarked against a large
  labeled test set - tune it based on your own query logs.

## Future Improvements

- Retrieval-Augmented Generation (RAG) over a larger document collection
- PDF/document ingestion for the knowledge base
- Multilingual support
- Voice input/output
- Admin dashboard for managing FAQs
- Authentication
- Usage analytics
- LLM-based response generation for genuinely open-ended queries
- A real vector database (FAISS, Chroma, or Pinecone) once the FAQ set grows
  past what fits comfortably in memory (see engineering notes below)

## Engineering Notes

- **Embeddings**: each sentence is converted into a 384-dimensional vector that
  captures meaning, not exact words - so paraphrases land close to their
  original question in vector space.
- **Cosine similarity**: measures the angle between two vectors, which turns
  out to be a much better semantic-relatedness signal than plain word overlap.
- **Why retrieval hallucinates less than generation**: this system can only
  ever return a pre-written answer from `faqs.json` - it has no ability to
  invent new text, so worst case it returns the *wrong* stored answer, never a
  fabricated one.
- **Scaling to 100,000+ FAQs**: this project brute-force compares the query
  against every FAQ embedding (`O(n)` per query), which is fine for dozens or
  low thousands of FAQs. Beyond that, you'd swap the in-memory NumPy array for
  an approximate-nearest-neighbor index (FAISS, Chroma, or a managed option
  like Pinecone) so lookups stay fast at scale without re-architecting the
  rest of the pipeline - the embedding + threshold logic stays identical.

## License

MIT - see [LICENSE](LICENSE).
