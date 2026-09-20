"""
test_api.py
-----------
API-level tests against the FastAPI app. Requires the model to actually
download on first run (needs internet the first time; cached after
that under ../models).

Run from the project root:
    pytest tests/test_api.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)  # triggers the lifespan -> loads the model once


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert "model_loaded" in res.json()


def test_languages_endpoint():
    res = client.get("/languages")
    assert res.status_code == 200
    names = [l["name"] for l in res.json()["languages"]]
    for expected in ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati"]:
        assert expected in names


def test_translate_english_to_hindi():
    res = client.post("/translate", json={
        "text": "Where are you going?",
        "source_language": "English",
        "target_language": "Hindi",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["translated_text"].strip() != ""
    assert data["source_language"] == "English"
    assert data["target_language"] == "Hindi"


def test_translate_hindi_to_english():
    res = client.post("/translate", json={
        "text": "आप कहाँ जा रहे हैं?",
        "source_language": "Hindi",
        "target_language": "English",
    })
    assert res.status_code == 200
    assert res.json()["translated_text"].strip() != ""


def test_translate_english_to_bengali():
    res = client.post("/translate", json={
        "text": "Good morning, how are you?",
        "source_language": "English",
        "target_language": "Bengali",
    })
    assert res.status_code == 200
    assert res.json()["translated_text"].strip() != ""


def test_empty_input_rejected():
    res = client.post("/translate", json={
        "text": "",
        "source_language": "English",
        "target_language": "Hindi",
    })
    assert res.status_code == 400


def test_unsupported_language_rejected():
    res = client.post("/translate", json={
        "text": "Hello",
        "source_language": "English",
        "target_language": "Klingon",
    })
    assert res.status_code == 400


def test_long_text_is_chunked_and_succeeds():
    long_text = " ".join([
        "This is a sentence used to build a long passage for testing."
    ] * 30)
    res = client.post("/translate", json={
        "text": long_text,
        "source_language": "English",
        "target_language": "Hindi",
    })
    assert res.status_code == 200
    assert res.json()["chunks_used"] >= 2


def test_history_flow():
    client.delete("/history")
    client.post("/translate", json={
        "text": "Thank you very much.",
        "source_language": "English",
        "target_language": "Hindi",
    })
    res = client.get("/history")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 1
    assert data["records"][-1]["source_language"] == "English"
