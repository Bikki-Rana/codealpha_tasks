"""
history.py
----------
Lightweight local translation history, stored as JSON on disk.

A real database (SQLite/Postgres) was deliberately NOT introduced here:
this is a single-user local tool, records are small, and a flat JSON
file is trivially readable/debuggable and needs zero setup. This is the
"do not introduce a database unless it genuinely improves the project"
guidance in practice.
"""

import json
import os
import threading
from datetime import datetime, timezone

from config import HISTORY_FILE, HISTORY_MAX_RECORDS

_lock = threading.Lock()


def _read_all() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A corrupted/partial history file should never crash the app -
        # treat it as empty and let new writes recover it.
        return []


def _write_all(records: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def add_record(source_language: str, target_language: str,
               source_text: str, translated_text: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_language": source_language,
        "target_language": target_language,
        "source_text": source_text,
        "translated_text": translated_text,
    }
    with _lock:
        records = _read_all()
        records.append(record)
        # keep the file bounded - drop oldest first
        if len(records) > HISTORY_MAX_RECORDS:
            records = records[-HISTORY_MAX_RECORDS:]
        _write_all(records)


def get_history() -> list[dict]:
    with _lock:
        return _read_all()


def clear_history() -> None:
    with _lock:
        _write_all([])
