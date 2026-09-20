"""
test_validation.py
-------------------
Focused tests for input validation / edge cases, independent of whether
the model itself produces "good" translations.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
from main import app
from config import MAX_CHARS

client = TestClient(app)


def test_whitespace_only_input_rejected():
    res = client.post("/translate", json={
        "text": "     ",
        "source_language": "English",
        "target_language": "Hindi",
    })
    assert res.status_code == 400


def test_same_source_and_target_rejected():
    res = client.post("/translate", json={
        "text": "Hello",
        "source_language": "English",
        "target_language": "English",
    })
    assert res.status_code == 400


def test_text_over_max_chars_rejected():
    too_long = "a" * (MAX_CHARS + 1)
    res = client.post("/translate", json={
        "text": too_long,
        "source_language": "English",
        "target_language": "Hindi",
    })
    assert res.status_code == 400


def test_malformed_request_missing_field():
    res = client.post("/translate", json={"text": "Hello"})
    assert res.status_code == 422  # FastAPI/Pydantic validation error


def test_invalid_json_type_rejected():
    res = client.post("/translate", json={
        "text": 12345,  # should be a string
        "source_language": "English",
        "target_language": "Hindi",
    })
    assert res.status_code == 422
