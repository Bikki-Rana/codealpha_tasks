import os

# Base directory = CodeAlpha_FAQChatbot/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Model ---
MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight, 384-dim, fast on CPU

# --- Data ---
FAQ_DATA_PATH = os.path.join(BASE_DIR, "data", "faqs.json")

# --- Retrieval ---
# Cosine similarity threshold below which we say "I don't know" (wired up in Phase 6).
# 0.5-0.6 is a reasonable starting point for MiniLM sentence similarity; we'll
# tune this empirically once we see real query scores.
CONFIDENCE_THRESHOLD = 0.55
