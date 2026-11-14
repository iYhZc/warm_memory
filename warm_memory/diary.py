"""Short daily diaries that also tidy duplicate memories.

Design notes:

* The diary never sees the raw conversation. It only sees a digest: at most
  ``diary_digest_max_chars`` characters, one shortened line per turn. That
  keeps the extra model call cheap and keeps raw chat text out of the prompt.
* Digest rows are deleted as soon as the diary that consumed them exists, so
  the only long-lived artifact is the diary itself.
* Numbering is per scope and starts at 1, matching ``/日记 查看 1``.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from .domain import DiaryEntry, DiaryTurn, ScopeKey
from .prompts import build_diary_prompt, build_diary_rewrite_prompt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import WarmMemoryConfig
    from .repository import SQLiteRepository

_DIGEST_USER_CHARS = 120
_DIGEST_BOT_CHARS = 80
_DIARY_PREVIEW_CHARS = 22
DIARY_SOURCE_MAX_CHARS = 12000
"""How much day material to retain for a later rewrite (the write prompt is smaller)."""
_FENCE_PATTERN = re.compile(r"```(?:json)?", re.IGNORECASE)
_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
_EMPTY_FEELING = "（今天没留下什么特别的感受）"
_DEFAULT_CLOCK = (23, 0)


def resolve_timezone(name: str) -> timezone | ZoneInfo:
    """Return the configured zone, or the host zone when tzdata is missing.

    A slim container image often has no ``/usr/share/zoneinfo``; falling back
    locally keeps the scheduler working instead of failing at import time.
    """
    if name:
        try:
            return ZoneInfo(name)
        except Exception:  # ZoneInfoNotFoundError and malformed names alike
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def parse_clock(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?", str(text or "").strip().replace("：", ":"))
    if not match:
        return _DEFAULT_CLOCK
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    if hour > 23 or minute > 59:
        return _DEFAULT_CLOCK
    return hour, minute


def is_due(now: datetime, hour: int, minute: int) -> bool:
    """True once today's scheduled minute has passed (catches up after restarts)."""
    return (now.hour, now.minute) >= (hour, minute)


def shorten(text: str, limit: int) -> str:
    """Flatten whitespace and cut to ``limit`` characters."""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    cut = compact[: max(0, int(limit) - 1)].rstrip("，,、；;：: ")
    return f"{cut}…" if cut else compact[: max(0, int(limit))]


def build_material(
    turns: Sequence[DiaryTurn],
    *,
    max_chars: int,
    user_chars: int = _DIGEST_USER_CHARS,
    bot_chars: int = _DIGEST_BOT_CHARS,
) -> str:
    """Turn stored digests into the diary prompt payload, never the raw chat."""
    lines: list[str] = []
    used = 0
    for turn in turns:
        user_text = shorten(turn.user_text, user_chars)
        if not user_text:
            continue
        line = f"- 对方：{user_text}"
        bot_text = shorten(turn.bot_text, bot_chars)
        if bot_text:
            line += f"\n  我：{bot_text}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_text(text: str) -> str:
    return _BLANK_LINES_PATTERN.sub("\n\n", str(text or "").strip())


def _clamp(text: str, limit: int | None) -> str:
    compact = _clean_text(text)
    if limit is None or len(compact) <= limit:
        return compact
    cut = compact[: max(0, limit - 1)].rstrip("，,、；;：: \n")
    return cut + "…" if cut else compact[:limit]


def parse_diary_output(
    raw: str,
    *,
    max_chars: int | None = None,
    feeling_max_chars: int | None = None,
) -> tuple[str, str]:
    """Require a valid ``{content, feeling}`` object before anything is stored.

    Both limits default to ``None``: diaries are no longer hard-truncated, so a
    sentence is never cut in half. Callers that really need a ceiling can still
    pass one explicitly.
    """
    text = _FENCE_PATTERN.sub("", str(raw or "")).strip()
    data = _json_object(text)
    if data is None:
        return "", ""
    content, feeling = data.get("content"), data.get("feeling", "")
    if not isinstance(content, str) or not isinstance(feeling, str) or not content.strip():
        return "", ""
    return _clamp(content, max_chars), _clamp(feeling, feeling_max_chars)


def format_cn_date(day: str) -> str:
    """``2026-11-01`` → ``2026年11月1日`` (no zero padding, as the user asked)."""
    parts = str(day or "").split("-")
    if len(parts) != 3:
        return str(day or "")
    try:
        year, month, date = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return str(day or "")
    return f"{year}年{month}月{date}日"


class DiaryService:
    """Own the diary lifecycle: digest → prompt → entry → tidy-up."""

    def __init__(
        self,
        repository: SQLiteRepository,
        config: WarmMemoryConfig,
        generate: Callable[[str, str | None], Awaitable[str]],
    ) -> None:
        self.repository = repository
        self.config = config
        self._generate = generate
        self.timezone = resolve_timezone(config.diary_timezone)

    # ------------------------------------------------------------------ clock

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    def today(self, now: datetime | None = None) -> str:
        return (now or self.now()).strftime("%Y-%m-%d")

    def due(self, now: datetime | None = None) -> bool:
        hour, minute = parse_clock(self.config.diary_time)
        return is_due(now or self.now(), hour, minute)

    # ----------------------------------------------------------------- digest

    async def record_turn(
        self,
        scope: ScopeKey,
        umo: str,
        user_text: str,
        bot_text: str,
        now: datetime | None = None,
    ) -> None:
        """Store one shortened turn; callers skip this when the session is off."""
        user_digest = shorten(user_text, _DIGEST_USER_CHARS)
        if not user_digest:
            return
        await asyncio.to_thread(
            self.repository.save_diary_turn,
            scope,
            umo,
            self.today(now),
            user_digest,
            shorten(bot_text, _DIGEST_BOT_CHARS),
        )

    # ------------------------------------------------------------------ write

    async def write(
        self,
        scope: ScopeKey,
        *,
        day: str | None = None,
        provider_id: str | None = None,
    ) -> DiaryEntry | None:
        """Generate one entry for ``day``; ``None`` means "nothing worth writing".

        One entry per day per scope is guaranteed by a unique index, so a second
        run for the same day is a no-op rather than a duplicate.
        """
        target_day = day or self.today()
        if await asyncio.to_thread(self.repository.has_diary, scope, target_day):
            return None
        turns = await asyncio.to_thread(self.repository.list_diary_turns, scope, target_day)
        if len(turns) < self.config.diary_min_turns:
            return None
        material = build_material(turns, max_chars=self.config.diary_digest_max_chars)
        if not material:
            return None
        # Keep a fuller day digest for later rewrites. The initial write still
        # uses the smaller, cheaper ``material`` above; this copy is not sent to
        # the model unless the user explicitly asks to rewrite the entry.
        source_material = build_material(
            turns,
            max_chars=max(self.config.diary_digest_max_chars, DIARY_SOURCE_MAX_CHARS),
        )
        prompt = build_diary_prompt(
            day=format_cn_date(target_day),
            material=material,
            custom_prompt=self.config.diary_prompt,
        )
        raw = await self._generate(prompt, provider_id)
        content, feeling = parse_diary_output(raw)
        if not content:
            raise ValueError("日记生成结果格式无效，素材已保留")
        entry = await asyncio.to_thread(
            self.repository.insert_diary,
            scope,
            target_day,
            content,
            feeling,
            source_material=source_material,
        )
        await asyncio.to_thread(
            self.repository.delete_diary_turns, scope, target_day,
            through_id=max((turn.row_id or 0 for turn in turns), default=0),
        )
        return entry

    async def rewrite(
        self,
        scope: ScopeKey,
        *,
        day: str | None = None,
        seq: int = 0,
        provider_id: str | None = None,
        fallback_material: str = "",
    ) -> DiaryEntry | None:
        """Rewrite an existing entry in place, with no character budget.

        ``seq`` selects an entry by its per-scope number; otherwise ``day``
        selects the calendar day, and when neither is given the latest entry is
        used. The source digest saved at write time is the source of truth; a
        caller may pass ``fallback_material`` (for example, extracted from the
        session history) for entries written before that column existed. If
        neither exists, the prompt only lets the model finish the existing text
        and forbids inventing facts.
        """
        if seq > 0:
            entry = await asyncio.to_thread(self.repository.get_diary, scope, int(seq))
        elif day:
            entry = await asyncio.to_thread(self.repository.get_diary_by_day, scope, day)
        else:
            entry = await asyncio.to_thread(self.repository.latest_diary, scope)
        if entry is None:
            return None
        prompt = build_diary_rewrite_prompt(
            day=format_cn_date(entry.day),
            content=entry.content,
            feeling=entry.feeling,
            material=entry.source_material or fallback_material,
            custom_prompt=self.config.diary_rewrite_prompt,
        )
        raw = await self._generate(prompt, provider_id)
        content, feeling = parse_diary_output(raw)
        if not content:
            return None
        return await asyncio.to_thread(
            self.repository.update_diary,
            scope,
            entry.id,
            content,
            feeling,
        )

    def resolve_day(self, text: str, now: datetime | None = None) -> str | None:
        """Turn "昨天"/"前天"/an ISO date into a local ``YYYY-MM-DD`` day.

        An empty value means yesterday, matching the conversational request
        "把昨天的日记重写一下". Unknown wording returns ``None`` instead of
        silently rewriting the wrong day.
        """
        value = str(text or "").strip()
        base = (now or self.now()).date()
        offsets = {
            "": 1,
            "昨天": 1,
            "昨日": 1,
            "今天": 0,
            "今日": 0,
            "前天": 2,
            "大前天": 3,
        }
        if value in offsets:
            return (base - timedelta(days=offsets[value])).isoformat()
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return None

    async def prune(self) -> int:
        """Trim leftover material to ``diary_material_keep_days`` days, counting today.

        Only ``diary_turns`` is touched: a written diary (``diaries``) is never
        pruned. Material from an earlier day can never be consumed anyway (the
        diary only ever reads ``day == today``), so this is pure housekeeping.
        Today's rows are always kept, because the diary may not be written yet.
        """
        keep = max(0, self.config.diary_material_keep_days)
        days_back = max(0, keep - 1)
        cutoff = (self.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        return await asyncio.to_thread(self.repository.prune_diary_turns, cutoff)

    # ------------------------------------------------------------------- read

    async def entries(self, scope: ScopeKey, *, limit: int, offset: int, oldest_first: bool):
        return await asyncio.to_thread(
            self.repository.list_diaries,
            scope,
            limit,
            offset,
            oldest_first=oldest_first,
        )

    async def count(self, scope: ScopeKey) -> int:
        return await asyncio.to_thread(self.repository.count_diaries, scope)

    async def find(self, scope: ScopeKey, seq: int) -> DiaryEntry | None:
        if seq <= 0:
            return await asyncio.to_thread(self.repository.latest_diary, scope)
        return await asyncio.to_thread(self.repository.get_diary, scope, seq)

    # ----------------------------------------------------------------- format

    def format_view(self, entry: DiaryEntry) -> str:
        """Date first, body in the middle, the bot's feeling at the bottom."""
        feeling = entry.feeling.strip() or _EMPTY_FEELING
        return (
            f"📔 日记 {entry.seq} · {format_cn_date(entry.day)}\n\n"
            f"{entry.content}\n\n"
            f"—— 感受 ——\n{feeling}"
        )

    def format_list(
        self,
        entries: Sequence[DiaryEntry],
        *,
        total: int,
        page: int,
        pages: int,
    ) -> str:
        lines = []
        for entry in entries:
            preview = shorten(entry.content.replace("\n", " "), _DIARY_PREVIEW_CHARS)
            lines.append(f"日记{entry.seq} · {format_cn_date(entry.day)} · {preview}")
        header = f"日记共 {total} 篇（第 {page}/{pages} 页）："
        footer = "" if pages <= 1 else "\n发送 /日记 列表 <页码> 翻页。"
        return header + "\n" + "\n".join(lines) + footer


__all__ = [
    "DIARY_SOURCE_MAX_CHARS",
    "DiaryService",
    "build_material",
    "format_cn_date",
    "is_due",
    "parse_clock",
    "parse_diary_output",
    "resolve_timezone",
    "shorten",
]
