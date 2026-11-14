"""Parsing and validation for reviewable manual-compression drafts."""

from __future__ import annotations

import json
from collections.abc import Sequence

from .memory_service import is_safe_memory_text

_ALLOWED_CATEGORIES = {
    "fact",
    "preference",
    "relationship",
    "commitment",
    "project",
    "event",
    "other",
}


def _json_object(raw: object) -> dict | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _optional_id(value: object, known_ids: frozenset[int] | None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    if known_ids is not None and value not in known_ids:
        return None
    return value


def parse_manual_compression(
    raw: str,
    *,
    max_memories: int = 3,
    max_summary_chars: int = 600,
    known_ids: Sequence[int] | None = None,
) -> tuple[str, list[dict]] | None:
    """Return ``(summary, candidates)`` or None when the model output is unusable."""
    payload = _json_object(raw)
    if payload is None:
        return None
    summary = payload.get("summary")
    if not isinstance(summary, str):
        return None
    summary = summary.strip()
    if not summary or len(summary) > max_summary_chars:
        return None

    items = payload.get("memories")
    if not isinstance(items, list):
        return None
    limit = min(max(int(max_memories), 1), 5)
    allowed_ids = frozenset(known_ids) if known_ids is not None else None
    candidates: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not is_safe_memory_text(content):
            continue
        normalized = "".join(content.split()).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        category = str(item.get("category", "other")).strip().lower()
        if category not in _ALLOWED_CATEGORIES:
            category = "other"
        try:
            importance = int(item.get("importance", 3))
        except (TypeError, ValueError):
            importance = 3
        importance = min(max(importance, 1), 5)
        try:
            confidence = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        confidence = min(max(confidence, 0.0), 1.0)
        candidates.append(
            {
                "ordinal": len(candidates) + 1,
                "content": content,
                "category": category,
                "importance": importance,
                "confidence": confidence,
                "status": "pending",
                "memory_id": None,
                "duplicate_of": _optional_id(item.get("duplicate_of"), allowed_ids),
                "replaces_id": _optional_id(item.get("replaces_id"), allowed_ids),
            }
        )
        if len(candidates) >= limit:
            break
    return summary, candidates
