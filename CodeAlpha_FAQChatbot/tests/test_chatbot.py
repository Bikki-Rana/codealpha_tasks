"""
Run with:  pytest tests/test_chatbot.py -v

Note: these tests spin up the real FAQSemanticSearch engine (via FastAPI's
startup event), which downloads/loads the MiniLM model on first run - the
first test session will be slower and needs internet access once.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers the startup event (model + FAQ loading)
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_exact_faq_question(client):
    response = client.post("/chat", json={"message": "What documents are required for admission?"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Admissions"
    assert data["confidence"] > 0.8


def test_paraphrased_question_matches_same_faq(client):
    response = client.post("/chat", json={"message": "Which papers do I need to join?"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Admissions"
    assert data["confidence"] > 0.5


def test_unrelated_question_is_low_confidence(client):
    response = client.post("/chat", json={"message": "What's the weather like today?"})
    assert response.status_code == 200
    data = response.json()
    # Either the confidence gate rejects it (category None) or, at minimum,
    # its score should sit noticeably below a real FAQ match.
    assert data["category"] is None or data["confidence"] < 0.6


def test_empty_input_is_handled_gracefully(client):
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["confidence"] == 0.0
    assert data["answer"]  # some prompt-for-input message, not empty/crash


def test_low_confidence_returns_fallback_message(client):
    response = client.post("/chat", json={"message": "asdkjqwoeiqwe random gibberish text zzz"})
    assert response.status_code == 200
    data = response.json()
    if data["confidence"] < 0.55:
        assert "couldn't find" in data["answer"].lower()


def test_faqs_endpoint_returns_list(client):
    response = client.get("/faqs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "question" in data[0] and "category" in data[0]
    assert "answer" not in data[0]  # /faqs should not leak internal fields unnecessarily


def test_clear_endpoint_resets_history(client):
    session_id = "test-session-123"
    client.post("/chat", json={"message": "What are the library working hours?", "session_id": session_id})
    response = client.post("/clear", json={"session_id": session_id})
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"
