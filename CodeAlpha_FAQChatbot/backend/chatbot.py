"""
Ties the semantic search engine to conversation-level behavior:
- applies the confidence threshold (unknown-question handling)
- keeps a per-session in-memory conversation history

Note on "conversation history" (see README Limitations): this system is
retrieval-based, not a generative LLM. Each question is still matched
independently against the FAQ set - history is stored and returned so the
UI can show a running chat log, but it is NOT used to help interpret
follow-up questions (e.g. "what about food?" after asking about hostel fees
will NOT automatically inherit the "hostel" context). Doing that properly
would require a query-rewriting step, which is out of scope for a pure
retrieval system without an LLM.
"""

try:
    from . import config
    from .embeddings import FAQSemanticSearch
except ImportError:
    import config
    from embeddings import FAQSemanticSearch

UNKNOWN_ANSWER = (
    "I'm sorry, I couldn't find a reliable answer in my FAQ database. "
    "Try asking about admissions, fees, exams, hostel, library, or placements."
)


class FAQChatbot:
    def __init__(self):
        self.engine = FAQSemanticSearch()
        self.sessions: dict[str, list[dict]] = {}

    def _history_for(self, session_id: str) -> list:
        return self.sessions.setdefault(session_id, [])

    def ask(self, message: str, session_id: str = "default") -> dict:
        message = (message or "").strip()
        history = self._history_for(session_id)

        if not message:
            response = {
                "answer": "Please type a question so I can help you.",
                "category": None,
                "confidence": 0.0,
                "matched_question": None,
            }
            history.append({"role": "user", "message": message})
            history.append({"role": "bot", "message": response["answer"]})
            return response

        result = self.engine.search(message)

        # --- Confidence threshold: this is what stops the bot from ever
        # confidently making up an answer to something outside the FAQ set. ---
        if result["confidence"] < config.CONFIDENCE_THRESHOLD:
            response = {
                "answer": UNKNOWN_ANSWER,
                "category": None,
                "confidence": result["confidence"],
                "matched_question": None,
            }
        else:
            response = {
                "answer": result["answer"],
                "category": result["category"],
                "confidence": result["confidence"],
                "matched_question": result["question_matched"],
            }

        history.append({"role": "user", "message": message})
        history.append({"role": "bot", "message": response["answer"]})
        return response

    def get_history(self, session_id: str = "default") -> list:
        return self._history_for(session_id)

    def clear_history(self, session_id: str = "default") -> None:
        self.sessions[session_id] = []
