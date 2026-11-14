"""Synchronous SQLite persistence; callers must use ``asyncio.to_thread``."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from .domain import (
    CaptureMessage,
    DiaryEntry,
    DiaryTurn,
    MemoryCandidate,
    MemoryRecord,
    ScopeKey,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK(scope_kind IN ('dm', 'group_shared', 'group_member')),
    group_id TEXT NOT NULL DEFAULT '',
    owner_id TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL CHECK(importance BETWEEN 1 AND 5),
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    source_user_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_recalled_at REAL,
    last_mentioned_at REAL,
    recall_count INTEGER NOT NULL DEFAULT 0,
    mention_count INTEGER NOT NULL DEFAULT 1,
    pinned INTEGER NOT NULL DEFAULT 0 CHECK(pinned IN (0, 1)),
    archived INTEGER NOT NULL DEFAULT 0 CHECK(archived IN (0, 1)),
    superseded_by INTEGER,
    valid_until REAL,
    source_quote TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS memories_scope_idx
    ON memories(bot_id, platform, scope_kind, group_id, owner_id, archived);
CREATE TABLE IF NOT EXISTS conversation_summaries (
    bot_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    umo TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    updated_at REAL NOT NULL,
    compressed_at REAL NOT NULL,
    mode TEXT NOT NULL,
    estimated_before_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_after_tokens INTEGER NOT NULL DEFAULT 0,
    anchor_hash TEXT NOT NULL DEFAULT '',
    retained_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(bot_id, platform, umo, conversation_id)
);
CREATE TABLE IF NOT EXISTS compression_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    umo TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    candidates TEXT NOT NULL,
    summary TEXT NOT NULL,
    prefix_hash TEXT NOT NULL,
    cut_count INTEGER NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    UNIQUE(bot_id, platform, umo, conversation_id)
);
CREATE TABLE IF NOT EXISTS capture_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    scope_kind TEXT NOT NULL,
    group_id TEXT NOT NULL DEFAULT '',
    owner_id TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    source_user_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL CHECK(state IN ('pending', 'buffered', 'inflight')),
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS capture_event_scope_idx
    ON capture_messages(event_id, bot_id, platform, scope_kind, group_id, owner_id);
CREATE INDEX IF NOT EXISTS capture_scope_idx
    ON capture_messages(bot_id, platform, scope_kind, group_id, owner_id, state, id);
CREATE INDEX IF NOT EXISTS capture_state_idx ON capture_messages(state, created_at);
CREATE TABLE IF NOT EXISTS diaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK(scope_kind IN ('dm', 'group_shared')),
    group_id TEXT NOT NULL DEFAULT '',
    owner_id TEXT NOT NULL DEFAULT '',
    seq INTEGER NOT NULL,
    day TEXT NOT NULL,
    content TEXT NOT NULL,
    feeling TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    source_material TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS diaries_scope_seq_idx
    ON diaries(bot_id, platform, scope_kind, group_id, owner_id, seq);
CREATE UNIQUE INDEX IF NOT EXISTS diaries_scope_day_idx
    ON diaries(bot_id, platform, scope_kind, group_id, owner_id, day);
CREATE TABLE IF NOT EXISTS diary_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    scope_kind TEXT NOT NULL,
    group_id TEXT NOT NULL DEFAULT '',
    owner_id TEXT NOT NULL DEFAULT '',
    umo TEXT NOT NULL DEFAULT '',
    day TEXT NOT NULL,
    user_text TEXT NOT NULL,
    bot_text TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS diary_turns_scope_day_idx
    ON diary_turns(bot_id, platform, scope_kind, group_id, owner_id, day, id);
CREATE TABLE IF NOT EXISTS diary_sessions (
    bot_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK(scope_kind IN ('dm', 'group_shared')),
    group_id TEXT NOT NULL DEFAULT '',
    owner_id TEXT NOT NULL DEFAULT '',
    umo TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(bot_id, platform, scope_kind, group_id, owner_id)
);
-- Retired reflection storage is retained for schema12/data compatibility only.
CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK(scope_kind IN ('dm', 'group_shared', 'group_member')),
    group_id TEXT NOT NULL DEFAULT '',
    owner_id TEXT NOT NULL DEFAULT '',
    day TEXT NOT NULL,
    content TEXT NOT NULL,
    source_diary_seq INTEGER,
    created_at REAL NOT NULL,
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS reflections_scope_idx
    ON reflections(bot_id, platform, scope_kind, group_id, owner_id, id);
"""

_SCOPE_WHERE = "bot_id=? AND platform=? AND scope_kind=? AND group_id=? AND owner_id=?"


class SQLiteRepository:
    """Store facts and compact derived state without retaining raw chat logs."""

    SCHEMA_VERSION = 12

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = NORMAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(
        self,
        validity_resolver: Callable[[str, float], float | None] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            current_version = int(db.execute("PRAGMA user_version").fetchone()[0])
            if current_version > self.SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {current_version} is newer than supported {self.SCHEMA_VERSION}"
                )
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(_SCHEMA)
            added_validity = self._migrate(db, current_version)
            if added_validity and validity_resolver is not None:
                # Rows written before v5 have no window; derive one from their
                # own wording so a stale "明天有早八" stops being injected.
                self._backfill_validity(db, validity_resolver)
            # A row claimed by an extraction that never finished (plugin reload)
            # must go back to the queue instead of being lost.
            db.execute("UPDATE capture_messages SET state='buffered' WHERE state='inflight'")
            # PRAGMA does not accept bound parameters; the value is an int constant.
            db.execute(f"PRAGMA user_version = {int(self.SCHEMA_VERSION)}")

    @staticmethod
    def _backfill_validity(
        db: sqlite3.Connection,
        resolver: Callable[[str, float], float | None],
    ) -> None:
        rows = db.execute(
            "SELECT id, content, created_at FROM memories WHERE valid_until IS NULL"
        ).fetchall()
        updates: list[tuple[float, int]] = []
        for row in rows:
            try:
                resolved = resolver(str(row["content"]), float(row["created_at"]))
            except Exception:
                resolved = None
            if resolved is not None:
                updates.append((float(resolved), int(row["id"])))
        if updates:
            db.executemany("UPDATE memories SET valid_until=? WHERE id=?", updates)

    @staticmethod
    def _migrate(db: sqlite3.Connection, from_version: int) -> bool:
        """Apply additive migrations; existing rows and columns stay untouched.

        Returns True when the validity columns were just added, so the caller
        can backfill them once.
        """
        columns = {str(row["name"]) for row in db.execute("PRAGMA table_info(memories)")}
        added_validity = False
        if "superseded_by" not in columns:
            db.execute("ALTER TABLE memories ADD COLUMN superseded_by INTEGER")
        if "valid_until" not in columns:
            db.execute("ALTER TABLE memories ADD COLUMN valid_until REAL")
            added_validity = True
        if "source_quote" not in columns:
            db.execute("ALTER TABLE memories ADD COLUMN source_quote TEXT NOT NULL DEFAULT ''")
        if "last_mentioned_at" not in columns:
            db.execute("ALTER TABLE memories ADD COLUMN last_mentioned_at REAL")
        summary_columns = {
            str(row["name"]) for row in db.execute("PRAGMA table_info(conversation_summaries)")
        }
        if "anchor_hash" not in summary_columns:
            db.execute(
                "ALTER TABLE conversation_summaries ADD COLUMN anchor_hash TEXT NOT NULL DEFAULT ''"
            )
        if "retained_count" not in summary_columns:
            db.execute(
                "ALTER TABLE conversation_summaries ADD COLUMN retained_count INTEGER NOT NULL DEFAULT 0"
            )
        diary_columns = {
            str(row["name"]) for row in db.execute("PRAGMA table_info(diaries)")
        }
        if "source_material" not in diary_columns:
            db.execute(
                "ALTER TABLE diaries ADD COLUMN source_material TEXT NOT NULL DEFAULT ''"
            )
        if from_version < 2:
            # Older databases may keep unused legacy columns; they are no
            # longer read or written, so no destructive migration is needed.
            db.execute(
                "CREATE INDEX IF NOT EXISTS memories_scope_idx "
                "ON memories(bot_id, platform, scope_kind, group_id, owner_id, archived)"
            )
        if from_version < 3:
            # The capture queue arrived in v3: the table is created above, and
            # nothing needs to be rewritten for rows that predate it.
            db.execute(
                "CREATE INDEX IF NOT EXISTS capture_state_idx "
                "ON capture_messages(state, created_at)"
            )
        if from_version < 4:
            # Diaries are additive: existing memories and summaries keep their
            # rows, and every session starts with diary writing off.
            db.execute(
                "CREATE INDEX IF NOT EXISTS diary_turns_scope_day_idx "
                "ON diary_turns(bot_id, platform, scope_kind, group_id, owner_id, day, id)"
            )
        # v9 removes affinity/reply-policy/persona-experiment tables from new
        # databases. Existing databases keep those legacy tables untouched:
        # nothing reads them, and dropping them would delete user data.
        return added_validity

    @staticmethod
    def _scope_where(scope: ScopeKey) -> tuple[str, str, str, str, str]:
        return scope.storage_tuple()

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> MemoryRecord:
        scope = ScopeKey(
            bot_id=row["bot_id"],
            platform=row["platform"],
            kind=row["scope_kind"],
            group_id=row["group_id"],
            owner_id=row["owner_id"],
        )
        superseded = row["superseded_by"]
        valid_until = row["valid_until"]
        last_mentioned = row["last_mentioned_at"]
        return MemoryRecord(
            id=int(row["id"]),
            scope=scope,
            category=str(row["category"]),
            content=str(row["content"]),
            importance=int(row["importance"]),
            confidence=float(row["confidence"]),
            source_user_id=str(row["source_user_id"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_recalled_at=(
                float(row["last_recalled_at"]) if row["last_recalled_at"] is not None else None
            ),
            recall_count=int(row["recall_count"]),
            mention_count=int(row["mention_count"]),
            pinned=bool(row["pinned"]),
            archived=bool(row["archived"]),
            superseded_by=(int(superseded) if superseded is not None else None),
            valid_until=(float(valid_until) if valid_until is not None else None),
            source_quote=str(row["source_quote"] or ""),
            last_mentioned_at=(
                float(last_mentioned) if last_mentioned is not None else None
            ),
        )

    def list_memories(self, scope: ScopeKey, include_archived: bool = False) -> list[MemoryRecord]:
        where = _SCOPE_WHERE
        params: tuple[object, ...] = self._scope_where(scope)
        if not include_archived:
            where += " AND archived=0"
        with self._connection() as db:
            rows = db.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY id", params
            ).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def count_memories(self, scope: ScopeKey, include_archived: bool = False) -> int:
        where = _SCOPE_WHERE
        params: tuple[object, ...] = self._scope_where(scope)
        if not include_archived:
            where += " AND archived=0"
        with self._connection() as db:
            row = db.execute(
                f"SELECT COUNT(*) AS total FROM memories WHERE {where}", params
            ).fetchone()
        return int(row["total"]) if row else 0

    def get_memory(self, scope: ScopeKey, memory_id: int) -> MemoryRecord | None:
        with self._connection() as db:
            row = db.execute(
                f"SELECT * FROM memories WHERE id=? AND {_SCOPE_WHERE}",
                (memory_id, *self._scope_where(scope)),
            ).fetchone()
        return self._memory_from_row(row) if row else None

    def insert_memory(
        self,
        scope: ScopeKey,
        candidate: MemoryCandidate,
        source_user_id: str,
        now: float | None = None,
    ) -> MemoryRecord:
        timestamp = time.time() if now is None else now
        with self._connection() as db:
            cursor = db.execute(
                """INSERT INTO memories (
                       bot_id, platform, scope_kind, group_id, owner_id,
                       category, content, importance, confidence, source_user_id,
                       created_at, updated_at, valid_until, source_quote
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    *self._scope_where(scope),
                    candidate.category,
                    candidate.content,
                    candidate.importance,
                    candidate.confidence,
                    source_user_id,
                    timestamp,
                    timestamp,
                    candidate.valid_until,
                    candidate.source_quote[:200],
                ),
            )
            memory_id = int(cursor.lastrowid)
        record = self.get_memory(scope, memory_id)
        if record is None:
            raise RuntimeError("inserted memory could not be read back")
        return record

    def update_memory_content(
        self,
        scope: ScopeKey,
        memory_id: int,
        content: str,
        now: float | None = None,
    ) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories SET content=?, updated_at=?
                    WHERE id=? AND {_SCOPE_WHERE}""",
                (
                    content,
                    time.time() if now is None else now,
                    memory_id,
                    *self._scope_where(scope),
                ),
            )
            return cursor.rowcount == 1

    def update_memory(
        self,
        scope: ScopeKey,
        memory_id: int,
        content: str,
        valid_until: float | None,
        now: float | None = None,
        *,
        source_quote: str | None = None,
    ) -> bool:
        """Rewrite a memory's text and validity window in one statement."""
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories SET content=?, valid_until=?, updated_at=?, source_quote=COALESCE(?, source_quote)
                    WHERE id=? AND {_SCOPE_WHERE}""",
                (
                    content,
                    valid_until,
                    time.time() if now is None else now,
                    source_quote,
                    memory_id,
                    *self._scope_where(scope),
                ),
            )
            return cursor.rowcount == 1

    def set_memory_pinned(
        self, scope: ScopeKey, memory_id: int, pinned: bool, now: float | None = None
    ) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories SET pinned=?, updated_at=?
                    WHERE id=? AND {_SCOPE_WHERE}""",
                (
                    int(pinned),
                    time.time() if now is None else now,
                    memory_id,
                    *self._scope_where(scope),
                ),
            )
            return cursor.rowcount == 1

    def increment_memory_mentions(
        self, scope: ScopeKey, memory_id: int, now: float | None = None
    ) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories SET mention_count=mention_count+1, updated_at=?
                    WHERE id=? AND {_SCOPE_WHERE}""",
                (time.time() if now is None else now, memory_id, *self._scope_where(scope)),
            )
            return cursor.rowcount == 1

    def reinforce_memory(
        self,
        scope: ScopeKey,
        memory_id: int,
        now: float | None = None,
        valid_until: float | None = None,
    ) -> bool:
        """Count a repeat mention, refresh decay, and optionally extend validity.

        A repeated fact is a new observation, so it should reset the decay clock
        just like a recall. If the candidate carries a later ``valid_until`` than
        an already expiring row, the existing row inherits it; an earlier or
        missing window never shortens the row's current expiry, and a permanent
        row stays permanent.
        """
        timestamp = time.time() if now is None else now
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories
                    SET mention_count=mention_count+1,
                        last_mentioned_at=?,
                        archived=0,
                        valid_until=CASE
                            WHEN ? IS NOT NULL
                                 AND valid_until IS NOT NULL
                                 AND valid_until < ?
                            THEN ? ELSE valid_until END,
                        updated_at=?
                    WHERE id=? AND {_SCOPE_WHERE} AND superseded_by IS NULL""",
                (
                    timestamp,
                    valid_until,
                    valid_until,
                    valid_until,
                    timestamp,
                    memory_id,
                    *self._scope_where(scope),
                ),
            )
            return cursor.rowcount == 1

    def add_memory_mentions(
        self,
        scope: ScopeKey,
        memory_id: int,
        count: int,
        now: float | None = None,
    ) -> bool:
        """Add several mentions at once when a fact is folded into another row."""
        if count <= 0:
            return False
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories SET mention_count=mention_count+?, updated_at=?
                    WHERE id=? AND {_SCOPE_WHERE}""",
                (count, time.time() if now is None else now, memory_id, *self._scope_where(scope)),
            )
            return cursor.rowcount == 1

    def record_recalls(
        self, scope: ScopeKey, memory_ids: Sequence[int], now: float | None = None
    ) -> None:
        if not memory_ids:
            return
        timestamp = time.time() if now is None else now
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connection() as db:
            db.execute(
                f"""UPDATE memories SET recall_count=recall_count+1, last_recalled_at=?, updated_at=?
                    WHERE id IN ({placeholders}) AND {_SCOPE_WHERE}""",
                (timestamp, timestamp, *memory_ids, *self._scope_where(scope)),
            )

    def archive_memories(
        self, scope: ScopeKey, memory_ids: Sequence[int], now: float | None = None
    ) -> int:
        """Archive a batch of decayed memories in a single statement."""
        if not memory_ids:
            return 0
        timestamp = time.time() if now is None else now
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories SET archived=1, updated_at=?
                    WHERE id IN ({placeholders}) AND {_SCOPE_WHERE} AND archived=0""",
                (timestamp, *memory_ids, *self._scope_where(scope)),
            )
            return cursor.rowcount

    def set_memory_archived(
        self, scope: ScopeKey, memory_id: int, archived: bool, now: float | None = None
    ) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories SET archived=?, updated_at=?
                    WHERE id=? AND {_SCOPE_WHERE}""",
                (
                    int(archived),
                    time.time() if now is None else now,
                    memory_id,
                    *self._scope_where(scope),
                ),
            )
            return cursor.rowcount == 1

    def restore_memory(
        self, scope: ScopeKey, memory_id: int, now: float | None = None
    ) -> bool:
        """Restore a row as an ordinary active memory.

        Clearing ``superseded_by`` matters: the user explicitly wants the old
        wording back, so it must no longer count as an old version that
        ``/记忆 维护`` may purge later.
        """
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories SET archived=0, superseded_by=NULL, updated_at=?
                    WHERE id=? AND {_SCOPE_WHERE}""",
                (
                    time.time() if now is None else now,
                    memory_id,
                    *self._scope_where(scope),
                ),
            )
            return cursor.rowcount == 1

    def supersede_memory(
        self,
        scope: ScopeKey,
        old_id: int,
        new_id: int,
        now: float | None = None,
    ) -> bool:
        """Retire ``old_id`` in favour of ``new_id`` without deleting its text."""
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE memories SET archived=1, superseded_by=?, updated_at=?
                    WHERE id=? AND {_SCOPE_WHERE}""",
                (new_id, time.time() if now is None else now, old_id, *self._scope_where(scope)),
            )
            return cursor.rowcount == 1

    def delete_memory(self, scope: ScopeKey, memory_id: int) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                f"DELETE FROM memories WHERE id=? AND {_SCOPE_WHERE}",
                (memory_id, *self._scope_where(scope)),
            )
            return cursor.rowcount == 1

    def clear_scope(self, scope: ScopeKey) -> int:
        with self._connection() as db:
            cursor = db.execute(
                f"DELETE FROM memories WHERE {_SCOPE_WHERE}",
                self._scope_where(scope),
            )
            return cursor.rowcount

    def get_summary(
        self, bot_id: str, platform: str, umo: str, conversation_id: str
    ) -> dict | None:
        with self._connection() as db:
            row = db.execute(
                """SELECT summary, updated_at, compressed_at, mode,
                          estimated_before_tokens, estimated_after_tokens,
                          anchor_hash, retained_count
                   FROM conversation_summaries
                   WHERE bot_id=? AND platform=? AND umo=? AND conversation_id=?""",
                (bot_id, platform, umo, conversation_id),
            ).fetchone()
        return dict(row) if row else None

    def save_summary(
        self,
        bot_id: str,
        platform: str,
        umo: str,
        conversation_id: str,
        summary: str,
        mode: str,
        estimated_before_tokens: int,
        estimated_after_tokens: int,
        now: float | None = None,
        anchor_hash: str = "",
        retained_count: int = 0,
    ) -> None:
        timestamp = time.time() if now is None else now
        with self._connection() as db:
            db.execute(
                """INSERT INTO conversation_summaries (
                       bot_id, platform, umo, conversation_id, summary, updated_at,
                       compressed_at, mode, estimated_before_tokens, estimated_after_tokens,
                       anchor_hash, retained_count
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(bot_id, platform, umo, conversation_id) DO UPDATE SET
                     summary=excluded.summary, updated_at=excluded.updated_at,
                     compressed_at=excluded.compressed_at, mode=excluded.mode,
                     estimated_before_tokens=excluded.estimated_before_tokens,
                     estimated_after_tokens=excluded.estimated_after_tokens,
                     anchor_hash=excluded.anchor_hash, retained_count=excluded.retained_count""",
                (
                    bot_id,
                    platform,
                    umo,
                    conversation_id,
                    summary,
                    timestamp,
                    timestamp,
                    mode,
                    max(0, estimated_before_tokens),
                    max(0, estimated_after_tokens),
                    anchor_hash,
                    max(0, int(retained_count)),
                ),
            )

    def clear_summary(self, bot_id: str, platform: str, umo: str, conversation_id: str) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                """DELETE FROM conversation_summaries
                   WHERE bot_id=? AND platform=? AND umo=? AND conversation_id=?""",
                (bot_id, platform, umo, conversation_id),
            )
            return cursor.rowcount == 1

    # ------------------------------------------------------- compression draft

    def save_compression_draft(
        self,
        bot_id: str,
        platform: str,
        umo: str,
        conversation_id: str,
        candidates: Sequence[dict],
        summary: str,
        prefix_hash: str,
        cut_count: int,
        now: float | None = None,
        expires_at: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        expiry = timestamp + 3 * 86400 if expires_at is None else expires_at
        payload = json.dumps(list(candidates), ensure_ascii=False, separators=(",", ":"))
        with self._connection() as db:
            db.execute(
                """INSERT INTO compression_drafts (
                       bot_id, platform, umo, conversation_id, candidates, summary,
                       prefix_hash, cut_count, created_at, expires_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(bot_id, platform, umo, conversation_id) DO UPDATE SET
                     candidates=excluded.candidates, summary=excluded.summary,
                     prefix_hash=excluded.prefix_hash, cut_count=excluded.cut_count,
                     created_at=excluded.created_at, expires_at=excluded.expires_at""",
                (
                    bot_id,
                    platform,
                    umo,
                    conversation_id,
                    payload,
                    summary,
                    prefix_hash,
                    max(0, int(cut_count)),
                    timestamp,
                    expiry,
                ),
            )

    def get_compression_draft(
        self, bot_id: str, platform: str, umo: str, conversation_id: str
    ) -> dict | None:
        with self._connection() as db:
            row = db.execute(
                """SELECT candidates, summary, prefix_hash, cut_count,
                          created_at, expires_at
                   FROM compression_drafts
                   WHERE bot_id=? AND platform=? AND umo=? AND conversation_id=?""",
                (bot_id, platform, umo, conversation_id),
            ).fetchone()
        if not row:
            return None
        try:
            candidates = json.loads(str(row["candidates"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(candidates, list):
            return None
        result = dict(row)
        result["candidates"] = candidates
        return result

    def delete_compression_draft(
        self, bot_id: str, platform: str, umo: str, conversation_id: str
    ) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                """DELETE FROM compression_drafts
                   WHERE bot_id=? AND platform=? AND umo=? AND conversation_id=?""",
                (bot_id, platform, umo, conversation_id),
            )
            return cursor.rowcount == 1

    def prune_compression_drafts(self, now: float | None = None) -> int:
        timestamp = time.time() if now is None else now
        with self._connection() as db:
            cursor = db.execute(
                "DELETE FROM compression_drafts WHERE expires_at <= ?", (timestamp,)
            )
            return cursor.rowcount

    # ----------------------------------------------------------------- capture

    def save_capture(
        self,
        event_id: str,
        scopes: Sequence[ScopeKey],
        text: str,
        source_user_id: str,
        now: float | None = None,
    ) -> int:
        """Record one user message for every scope it belongs to."""
        timestamp = time.time() if now is None else now
        inserted = 0
        with self._connection() as db:
            for scope in scopes:
                cursor = db.execute(
                    """INSERT OR IGNORE INTO capture_messages (
                           event_id, bot_id, platform, scope_kind, group_id, owner_id,
                           text, source_user_id, state, attempts, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?)""",
                    (
                        event_id,
                        *self._scope_where(scope),
                        text,
                        source_user_id,
                        timestamp,
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def mark_capture_buffered(self, event_id: str) -> int:
        with self._connection() as db:
            cursor = db.execute(
                "UPDATE capture_messages SET state='buffered' WHERE event_id=? AND state='pending'",
                (event_id,),
            )
            return cursor.rowcount

    def delete_capture(self, event_id: str) -> int:
        with self._connection() as db:
            cursor = db.execute("DELETE FROM capture_messages WHERE event_id=?", (event_id,))
            return cursor.rowcount

    def delete_capture_ids(self, row_ids: Sequence[int]) -> int:
        if not row_ids:
            return 0
        placeholders = ",".join("?" for _ in row_ids)
        with self._connection() as db:
            cursor = db.execute(
                f"DELETE FROM capture_messages WHERE id IN ({placeholders})",
                tuple(row_ids),
            )
            return cursor.rowcount

    def release_capture_ids(self, row_ids: Sequence[int], max_attempts: int, *, failed_attempt: bool = True) -> int:
        """Return claimed rows to the queue; drop the ones out of retries."""
        if not row_ids:
            return 0
        placeholders = ",".join("?" for _ in row_ids)
        with self._connection() as db:
            db.execute(
                f"""UPDATE capture_messages SET state='buffered', attempts=attempts+?
                    WHERE id IN ({placeholders})""",
                (int(failed_attempt), *row_ids),
            )
            cursor = db.execute(
                f"DELETE FROM capture_messages WHERE id IN ({placeholders}) AND attempts > ?",
                (*row_ids, max(1, int(max_attempts))),
            )
            return cursor.rowcount

    def claim_capture_batch(self, batch_size: int) -> dict[ScopeKey, tuple[CaptureMessage, ...]]:
        """Claim up to ``batch_size`` buffered messages per scope, atomically."""
        size = max(1, int(batch_size))
        claimed: dict[ScopeKey, tuple[CaptureMessage, ...]] = {}
        with self._connection() as db:
            rows = db.execute(
                """SELECT id, bot_id, platform, scope_kind, group_id, owner_id,
                          text, source_user_id, created_at
                   FROM capture_messages WHERE state='buffered' ORDER BY id"""
            ).fetchall()
            grouped: dict[ScopeKey, list[sqlite3.Row]] = {}
            for row in rows:
                try:
                    scope = ScopeKey(
                        bot_id=str(row["bot_id"]),
                        platform=str(row["platform"]),
                        kind=str(row["scope_kind"]),
                        group_id=str(row["group_id"]),
                        owner_id=str(row["owner_id"]),
                    )
                except ValueError:
                    continue
                grouped.setdefault(scope, []).append(row)
            for scope, entries in grouped.items():
                if len(entries) < size:
                    continue
                chunk = entries[:size]
                row_ids = [int(entry["id"]) for entry in chunk]
                db.execute(
                    "UPDATE capture_messages SET state='inflight' "
                    f"WHERE id IN ({','.join('?' for _ in row_ids)})",
                    tuple(row_ids),
                )
                claimed[scope] = tuple(
                    CaptureMessage(
                        text=str(entry["text"]),
                        source_user_id=str(entry["source_user_id"]),
                        row_id=int(entry["id"]),
                        created_at=float(entry["created_at"]),
                    )
                    for entry in chunk
                )
        return claimed

    def count_capture(self, state: str) -> int:
        with self._connection() as db:
            row = db.execute(
                "SELECT COUNT(*) AS total FROM capture_messages WHERE state=?", (state,)
            ).fetchone()
        return int(row["total"]) if row else 0

    def prune_pending_capture(self, ttl_seconds: float, now: float | None = None) -> int:
        """Drop messages staged by turns that never completed within the TTL."""
        if ttl_seconds <= 0:
            return 0
        cutoff = (time.time() if now is None else now) - ttl_seconds
        with self._connection() as db:
            cursor = db.execute(
                "DELETE FROM capture_messages WHERE state='pending' AND created_at < ?",
                (cutoff,),
            )
            return cursor.rowcount

    def prune_all_pending_capture(self) -> int:
        with self._connection() as db:
            cursor = db.execute(
                "DELETE FROM capture_messages WHERE state='pending' AND created_at < ?",
                (time.time() - 3600,),
            )
            return cursor.rowcount

    def prune_superseded(self, retain_days: int = 90, now: float | None = None) -> int:
        """Physically drop old superseded rows that are still archived.

        A restored row is active and has no ``superseded_by`` marker, so it is
        never matched here even if its content once replaced another fact.
        """
        cutoff = (time.time() if now is None else now) - max(retain_days, 1) * 86400
        with self._connection() as db:
            cursor = db.execute(
                "DELETE FROM memories "
                "WHERE superseded_by IS NOT NULL AND archived=1 AND updated_at < ?",
                (cutoff,),
            )
            return cursor.rowcount

    def vacuum(self) -> None:
        with self._connection() as db:
            db.execute("VACUUM")

    # ------------------------------------------------------------------ diaries

    @staticmethod
    def _diary_from_row(row: sqlite3.Row) -> DiaryEntry:
        return DiaryEntry(
            id=int(row["id"]),
            scope=ScopeKey(
                bot_id=str(row["bot_id"]),
                platform=str(row["platform"]),
                kind=str(row["scope_kind"]),
                group_id=str(row["group_id"]),
                owner_id=str(row["owner_id"]),
            ),
            seq=int(row["seq"]),
            day=str(row["day"]),
            content=str(row["content"]),
            feeling=str(row["feeling"]),
            created_at=float(row["created_at"]),
            source_material=str(row["source_material"] or ""),
        )

    def get_diary(self, scope: ScopeKey, seq: int) -> DiaryEntry | None:
        with self._connection() as db:
            row = db.execute(
                f"SELECT * FROM diaries WHERE seq=? AND {_SCOPE_WHERE}",
                (int(seq), *self._scope_where(scope)),
            ).fetchone()
        return self._diary_from_row(row) if row else None

    def get_diary_by_day(self, scope: ScopeKey, day: str) -> DiaryEntry | None:
        """Return the one entry for a local calendar day, if it exists."""
        with self._connection() as db:
            row = db.execute(
                f"SELECT * FROM diaries WHERE day=? AND {_SCOPE_WHERE}",
                (str(day), *self._scope_where(scope)),
            ).fetchone()
        return self._diary_from_row(row) if row else None

    def latest_diary(self, scope: ScopeKey) -> DiaryEntry | None:
        with self._connection() as db:
            row = db.execute(
                f"SELECT * FROM diaries WHERE {_SCOPE_WHERE} ORDER BY seq DESC LIMIT 1",
                self._scope_where(scope),
            ).fetchone()
        return self._diary_from_row(row) if row else None

    def list_diaries(
        self,
        scope: ScopeKey,
        limit: int = 20,
        offset: int = 0,
        *,
        oldest_first: bool = False,
    ) -> list[DiaryEntry]:
        order = "ASC" if oldest_first else "DESC"
        with self._connection() as db:
            rows = db.execute(
                f"""SELECT * FROM diaries WHERE {_SCOPE_WHERE}
                    ORDER BY seq {order} LIMIT ? OFFSET ?""",
                (*self._scope_where(scope), max(1, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [self._diary_from_row(row) for row in rows]

    def count_diaries(self, scope: ScopeKey) -> int:
        with self._connection() as db:
            row = db.execute(
                f"SELECT COUNT(*) AS total FROM diaries WHERE {_SCOPE_WHERE}",
                self._scope_where(scope),
            ).fetchone()
        return int(row["total"]) if row else 0

    def has_diary(self, scope: ScopeKey, day: str) -> bool:
        with self._connection() as db:
            row = db.execute(
                f"SELECT 1 FROM diaries WHERE day=? AND {_SCOPE_WHERE}",
                (day, *self._scope_where(scope)),
            ).fetchone()
        return row is not None

    def insert_diary(
        self,
        scope: ScopeKey,
        day: str,
        content: str,
        feeling: str,
        now: float | None = None,
        source_material: str = "",
    ) -> DiaryEntry:
        """Append one entry; ``seq`` continues the per-scope 1-based numbering."""
        timestamp = time.time() if now is None else now
        scope_values = self._scope_where(scope)
        with self._connection() as db:
            cursor = db.execute(
                f"""INSERT INTO diaries (
                       bot_id, platform, scope_kind, group_id, owner_id,
                       seq, day, content, feeling, created_at, source_material
                   ) VALUES (
                       ?, ?, ?, ?, ?,
                       (SELECT COALESCE(MAX(seq), 0) + 1 FROM diaries WHERE {_SCOPE_WHERE}),
                       ?, ?, ?, ?, ?
                   )""",
                (
                    *scope_values,
                    *scope_values,
                    day,
                    content,
                    feeling,
                    timestamp,
                    source_material,
                ),
            )
            diary_id = int(cursor.lastrowid)
            row = db.execute("SELECT * FROM diaries WHERE id=?", (diary_id,)).fetchone()
        if row is None:
            raise RuntimeError("inserted diary could not be read back")
        return self._diary_from_row(row)

    def update_diary(
        self,
        scope: ScopeKey,
        diary_id: int,
        content: str,
        feeling: str,
        *,
        source_material: str | None = None,
    ) -> DiaryEntry | None:
        """Replace an entry's text in place, keeping its seq and day stable."""
        with self._connection() as db:
            cursor = db.execute(
                f"""UPDATE diaries
                    SET content=?,
                        feeling=?,
                        source_material=CASE
                            WHEN ? IS NULL THEN source_material ELSE ?
                        END
                    WHERE id=? AND {_SCOPE_WHERE}""",
                (
                    content,
                    feeling,
                    source_material,
                    source_material,
                    int(diary_id),
                    *self._scope_where(scope),
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = db.execute("SELECT * FROM diaries WHERE id=?", (int(diary_id),)).fetchone()
        return self._diary_from_row(row) if row else None

    # ------------------------------------------------------------- diary turns

    def save_diary_turn(
        self,
        scope: ScopeKey,
        umo: str,
        day: str,
        user_text: str,
        bot_text: str,
        now: float | None = None,
    ) -> None:
        """Keep one shortened turn; the raw message is never stored."""
        with self._connection() as db:
            db.execute(
                """INSERT INTO diary_turns (
                       bot_id, platform, scope_kind, group_id, owner_id,
                       umo, day, user_text, bot_text, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    *self._scope_where(scope),
                    umo,
                    day,
                    user_text,
                    bot_text,
                    time.time() if now is None else now,
                ),
            )

    def list_diary_turns(self, scope: ScopeKey, day: str) -> list[DiaryTurn]:
        with self._connection() as db:
            rows = db.execute(
                f"""SELECT id, user_text, bot_text, created_at FROM diary_turns
                    WHERE day=? AND {_SCOPE_WHERE} ORDER BY id""",
                (day, *self._scope_where(scope)),
            ).fetchall()
        return [
            DiaryTurn(
                user_text=str(row["user_text"]),
                bot_text=str(row["bot_text"]),
                created_at=float(row["created_at"]),
                row_id=int(row["id"]),
            )
            for row in rows
        ]

    def count_diary_turns(self, scope: ScopeKey, day: str) -> int:
        with self._connection() as db:
            row = db.execute(
                f"SELECT COUNT(*) AS total FROM diary_turns WHERE day=? AND {_SCOPE_WHERE}",
                (day, *self._scope_where(scope)),
            ).fetchone()
        return int(row["total"]) if row else 0

    def delete_diary_turns(self, scope: ScopeKey, day: str, *, through_id: int | None = None) -> int:
        """Consume a captured snapshot without deleting later arrivals."""
        boundary = " AND id <= ?" if through_id is not None else ""
        values = (day, *self._scope_where(scope))
        if through_id is not None:
            values += (through_id,)
        with self._connection() as db:
            cursor = db.execute(
                f"DELETE FROM diary_turns WHERE day=? AND {_SCOPE_WHERE}" + boundary,
                values,
            )
            return cursor.rowcount

    def prune_diary_turns(self, before_day: str) -> int:
        """Drop digests of days that no longer need to be kept."""
        with self._connection() as db:
            cursor = db.execute("DELETE FROM diary_turns WHERE day < ?", (before_day,))
            return cursor.rowcount

    # ----------------------------------------------------------- diary sessions

    def enable_diary_session(self, scope: ScopeKey, umo: str, now: float | None = None) -> None:
        with self._connection() as db:
            db.execute(
                """INSERT INTO diary_sessions (
                       bot_id, platform, scope_kind, group_id, owner_id, umo, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(bot_id, platform, scope_kind, group_id, owner_id)
                   DO UPDATE SET umo=excluded.umo, created_at=excluded.created_at""",
                (*self._scope_where(scope), umo, time.time() if now is None else now),
            )

    def disable_diary_session(self, scope: ScopeKey) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                f"DELETE FROM diary_sessions WHERE {_SCOPE_WHERE}", self._scope_where(scope)
            )
            return cursor.rowcount == 1

    def is_diary_session(self, scope: ScopeKey) -> bool:
        with self._connection() as db:
            row = db.execute(
                f"SELECT 1 FROM diary_sessions WHERE {_SCOPE_WHERE}",
                self._scope_where(scope),
            ).fetchone()
        return row is not None

    def list_diary_sessions(self) -> list[tuple[ScopeKey, str]]:
        with self._connection() as db:
            rows = db.execute("SELECT * FROM diary_sessions ORDER BY created_at").fetchall()
        return [
            (
                ScopeKey(
                    bot_id=str(row["bot_id"]),
                    platform=str(row["platform"]),
                    kind=str(row["scope_kind"]),
                    group_id=str(row["group_id"]),
                    owner_id=str(row["owner_id"]),
                ),
                str(row["umo"]),
            )
            for row in rows
        ]
