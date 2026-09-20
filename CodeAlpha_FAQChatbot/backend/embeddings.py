"""
Semantic search engine for the FAQ chatbot.

Pipeline:
    FAQ questions  --(encode once, at startup)-->  FAQ embeddings (cached in memory)
    User query     --(encode per request)-->        query embedding
    cosine_similarity(query embedding, all FAQ embeddings) -> best match + score
"""

import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from . import config
except ImportError:
    # Allows running this file directly (`python embeddings.py`) as well as
    # importing it as part of the `backend` package.
    import config


class FAQSemanticSearch:
    def __init__(self, faq_path: str = config.FAQ_DATA_PATH, model_name: str = config.MODEL_NAME):
        if not os.path.exists(faq_path):
            raise FileNotFoundError(
                f"[ERROR] FAQ data file not found at: {faq_path}. "
                f"Make sure data/faqs.json exists before starting the server."
            )

        print(f"[INFO] Loading embedding model: {model_name} ...")
        try:
            # Imported lazily so a missing/failed model download gives a clear
            # error instead of crashing at module import time.
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Could not load embedding model '{model_name}'. "
                f"Check your internet connection (first run downloads the model) "
                f"and that sentence-transformers is installed. Original error: {e}"
            ) from e
        print("[INFO] Model loaded.")

        print(f"[INFO] Loading FAQ data from {faq_path} ...")
        try:
            with open(faq_path, "r", encoding="utf-8") as f:
                self.faqs = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"[ERROR] faqs.json is not valid JSON: {e}") from e

        if not self.faqs:
            raise ValueError("[ERROR] faqs.json is empty - no FAQ entries to search over.")

        print(f"[INFO] Loaded {len(self.faqs)} FAQ entries.")

        self.questions = [faq["question"] for faq in self.faqs]

        print("[INFO] Encoding FAQ questions into embeddings...")
        # normalize_embeddings=True makes the vectors unit-length, so a plain
        # dot product equals cosine similarity - slightly cheaper to compute later.
        self.embeddings = self.model.encode(
            self.questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        print("[INFO] FAQ embeddings ready. Semantic search engine initialized.")

    def search(self, user_query: str) -> dict:
        """
        Encode the user's query and return the single best-matching FAQ,
        along with a cosine similarity score in [-1, 1] (in practice, [0, 1]
        for related sentences).
        """
        if not user_query or not user_query.strip():
            return {
                "question_matched": None,
                "answer": None,
                "category": None,
                "confidence": 0.0,
            }

        query_embedding = self.model.encode(
            [user_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_faq = self.faqs[best_idx]

        return {
            "question_matched": best_faq["question"],
            "answer": best_faq["answer"],
            "category": best_faq["category"],
            "confidence": round(best_score, 4),
        }


if __name__ == "__main__":
    # Quick manual smoke test - run with: python backend/embeddings.py
    engine = FAQSemanticSearch()

    test_queries = [
        "Which papers are required to join?",       # paraphrase of FAQ #1
        "Is food included in the hostel charges?",   # paraphrase of FAQ #29
        "What's the weather like today?",            # unrelated - should score low
    ]

    for q in test_queries:
        result = engine.search(q)
        print(f"\nQuery: {q}")
        print(f"  Matched FAQ : {result['question_matched']}")
        print(f"  Answer      : {result['answer']}")
        print(f"  Category    : {result['category']}")
        print(f"  Confidence  : {result['confidence']}")
