"""Small, content-free daily usage ledger, independent of the memory database."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import math
from pathlib import Path
import sqlite3
import time


_SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")
_COUNTERS = ("calls", "successes", "errors", "cancelled", "usage_known", "usage_unknown",
             "input_other", "input_cached", "output", "elapsed_ms")
_MISSING = object()


def _field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name, _MISSING)
    return getattr(value, name, _MISSING)


def _token_count(value: object) -> int | None:
    # Never silently coerce missing, bool, strings, negative or fractional counts.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return None
    if value < 0 or value > 2**63 - 1:
        return None
    return int(value)


def _tokens(usage: object | None) -> tuple[int, int, int] | None:
    native = tuple(_field(usage, name) for name in ("input_other", "input_cached", "output"))
    if any(value is not _MISSING for value in native):
        counts = tuple(_token_count(value) for value in native)
    else:
        prompt = _token_count(_field(usage, "prompt_tokens"))
        output = _token_count(_field(usage, "completion_tokens"))
        cached_value = _field(_field(usage, "prompt_tokens_details"), "cached_tokens")
        if cached_value is _MISSING:
            cached_value = _field(usage, "prompt_cache_hit_tokens")
        cached = _token_count(cached_value)
        if prompt is None or cached is None or output is None or cached > prompt:
            return None
        counts = (prompt - cached, cached, output)
    if any(value is None for value in counts):
        return None
    return counts


def _label(value: str, limit: int) -> str:
    # In particular, never stringify a provider config or response payload.
    if not isinstance(value, str):
        raise TypeError("usage labels must be strings")
    return "".join(char for char in value if char.isprintable())[:limit]


def _scope_hash(scope_key: str) -> str:
    if not isinstance(scope_key, str):
        raise TypeError("scope_key must be a string")
    return hashlib.sha256(scope_key.encode("utf-8")).hexdigest()


def _day(now: float | None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now, _SHANGHAI).date().isoformat()


class UsageLedger:
    """Thread-safe per-call connections; only hashed scopes and counters persist.

    ``elapsed`` is in seconds. Reports sum milliseconds and retain the distinction
    between measured zero usage and absent/invalid token usage. Failed/cancelled
    calls may still carry measured usage and are counted independently of status.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        columns = ", ".join(f"{name} INTEGER NOT NULL DEFAULT 0" for name in _COUNTERS)
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"""CREATE TABLE IF NOT EXISTS usage_daily (
                day TEXT NOT NULL, scope_hash TEXT NOT NULL,
                purpose TEXT NOT NULL, model TEXT NOT NULL,
                {columns}, PRIMARY KEY (day, scope_hash, purpose, model)
            )""")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=0.5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def record(self, scope_key: str, purpose: str, model: str, status: str,
               elapsed: float, usage: object | None = None, now: float | None = None) -> None:
        if status not in ("success", "error", "cancelled"):
            raise ValueError("usage status must be success, error or cancelled")
        scope = _scope_hash(scope_key)
        purpose, model = _label(purpose, 64), _label(model, 160)
        day = _day(now)
        cutoff = (datetime.fromisoformat(day).date() - timedelta(days=89)).isoformat()
        tokens = _tokens(usage)
        elapsed_ms = 0
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            if math.isfinite(elapsed) and 0 <= elapsed <= (2**63 - 1) / 1000:
                elapsed_ms = round(elapsed * 1000)
        counters = (1, int(status == "success"), int(status == "error"), int(status == "cancelled"),
                    int(tokens is not None), int(tokens is None), *(tokens or (0, 0, 0)), elapsed_ms)
        names = ", ".join(_COUNTERS)
        placeholders = ", ".join("?" for _ in range(4 + len(_COUNTERS)))
        update = ", ".join(f"{name} = {name} + excluded.{name}" for name in _COUNTERS)
        with closing(self._connect()) as conn, conn:
            conn.execute(f"""INSERT INTO usage_daily (day, scope_hash, purpose, model, {names})
                VALUES ({placeholders}) ON CONFLICT(day, scope_hash, purpose, model)
                DO UPDATE SET {update}""", (day, scope, purpose, model, *counters))
            # Only this independent, derived statistics table is subject to retention.
            conn.execute("DELETE FROM usage_daily WHERE day < ?", (cutoff,))

    def report(self, scope_key: str, days: int = 7, now: float | None = None) -> list[dict]:
        days = max(1, min(90, int(days)))
        day = _day(now)
        start = (datetime.fromisoformat(day).date() - timedelta(days=days - 1)).isoformat()
        sums = ", ".join(f"SUM({name}) AS {name}" for name in _COUNTERS)
        with closing(self._connect()) as conn:
            rows = conn.execute(f"""SELECT purpose, model, {sums} FROM usage_daily
                WHERE scope_hash = ? AND day >= ? AND day <= ?
                GROUP BY purpose, model ORDER BY purpose, model""",
                (_scope_hash(scope_key), start, day)).fetchall()
        return [dict(row) for row in rows]
