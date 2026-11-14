"""Parse the nightly memory-curation answer into validated operations.

The model sees the current memory list and may propose four kinds of edits:
``update`` an existing row, ``merge`` several rows into one, ``archive`` an
obsolete row, or ``add`` a durable fact that is missing. Everything is checked
against the ids that were actually shown, so a hallucinated id is dropped
instead of touching an unrelated memory.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace

from .domain import MemoryOp
from .memory_service import is_safe_memory_text

_CONTAINER = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_KINDS = {"update", "merge", "archive", "add"}


def _ids(value: object) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, list):
        result: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                continue
            result.append(item)
        return list(dict.fromkeys(result))
    return []


def _number(value: object, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return min(max(parsed, low), high)


def parse_memory_ops(
    raw: str,
    allowed_ids: frozenset[int],
    *,
    max_ops: int = 8,
    source_texts: Sequence[str] | None = None,
) -> list[MemoryOp]:
    """Read ``{"ops": [...]}`` and drop anything unsafe or out of scope."""
    text = _CONTAINER.sub("", str(raw or "")).strip()
    if not text:
        return []
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    items = payload.get("ops")
    if not isinstance(items, list):
        items = payload.get("memory_ops")
    if not isinstance(items, list):
        return []

    ops: list[MemoryOp] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("op") or item.get("kind") or "").strip().lower()
        if kind not in _KINDS:
            continue
        ids = tuple(_ids(item.get("ids") or item.get("id")))
        if any(memory_id not in allowed_ids for memory_id in ids):
            continue
        quote = str(item.get("source_quote") or "").strip()
        if source_texts is not None and (
            len(quote) < 4 or len(quote) > 600
            or not any(quote in text for text in source_texts)
        ):
            continue
        content = str(item.get("content") or "").strip()
        reason = str(item.get("reason") or "").strip()[:120]

        if kind == "archive":
            if len(ids) != 1:
                continue
            ops.append(MemoryOp(kind="archive", ids=ids, reason=reason))
        elif kind == "update":
            if len(ids) != 1 or not is_safe_memory_text(content):
                continue
            ops.append(MemoryOp(kind="update", ids=ids, content=content, reason=reason))
        elif kind == "merge":
            if len(ids) < 2 or not is_safe_memory_text(content):
                continue
            ops.append(MemoryOp(kind="merge", ids=ids, content=content, reason=reason))
        else:  # add
            if not is_safe_memory_text(content):
                continue
            ops.append(
                MemoryOp(
                    kind="add",
                    content=content,
                    category=str(item.get("category") or "other").strip().lower(),
                    importance=_number(item.get("importance"), 3, 1, 5),
                    reason=reason,
                )
            )
        ops[-1] = replace(ops[-1], source_quote=quote)
        if len(ops) >= max_ops:
            break
    return ops
