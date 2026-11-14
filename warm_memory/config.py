"""Validated plugin configuration and safe defaults."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .domain import ScopeKey


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return default


def _as_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _as_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _as_string(value: object, default: str = "") -> str:
    if not isinstance(value, str):
        return default
    return value.strip()


def _as_clock(value: object, default: str = "23:00") -> str:
    """Accept ``23``, ``23:00`` or ``23：00`` and return normalised ``HH:MM``."""
    if not isinstance(value, str):
        return default
    text = value.strip().replace("：", ":")
    if not re.fullmatch(r"\d{1,2}(?::\d{1,2})?", text):
        return default
    hours, _, minutes = text.partition(":")
    hour = int(hours)
    minute = int(minutes) if minutes else 0
    if hour > 23 or minute > 59:
        return default
    return f"{hour:02d}:{minute:02d}"


@dataclass(frozen=True, slots=True)
class WarmMemoryConfig:
    auto_memory_enabled: bool = True
    message_dedupe_seconds: float = 10.0
    """Drop re-deliveries of the same OneBot message id within this window (0 = off)."""
    chat_style_guard: bool = True
    """Append a persona-neutral "chat, not homework" block to the system prompt."""
    strip_reasoning_history: bool = True
    """Drop the model's own chain-of-thought from assistant history before a call."""
    memory_extract_interval: int = 12
    memory_extract_concurrency: int = 1
    auto_memory_min_importance: int = 3
    memory_capture_ttl_seconds: int = 900
    group_memory_enabled: bool = False
    min_saving_ratio: float = 0.30
    context_window_override: int = 0
    fallback_context_tokens: int = 128000
    keep_recent_turns: int = 8
    summary_model_id: str = ""
    summary_max_chars: int = 600
    memory_recall_limit: int = 5
    memory_recall_max_chars: int = 800
    memory_context_placement: str = "system"
    memory_recall_min_relevance: float = 0.15
    memory_recall_always_include_important: bool = False
    memory_recall_always_top_k: int = 0
    memory_recall_fallback: bool = False
    memory_retrieval_timeout: float = 8.0
    memory_semantic_enabled: bool = True
    memory_semantic_timeout: float = 6.0
    memory_semantic_model_id: str = ""
    memory_embedding_enabled: bool = False
    memory_embedding_model_id: str = ""
    memory_embedding_timeout: float = 6.0
    memory_embedding_min_similarity: float = 0.55
    memory_rerank_model_id: str = ""
    memory_rerank_timeout: float = 4.0
    memory_rerank_min_score: float = 0.0
    owner_alias: str = ""
    """Only used for private scopes matching the complete owner identity."""
    owner_platform: str = ""
    owner_bot_id: str = ""
    owner_user_id: str = ""
    memory_dedupe_context: int = 40
    """How many active memories the extractor sees for duplicate detection."""
    base_half_life_days: int = 30
    archive_retention_threshold: float = 0.10
    destructive_commands_require_confirm: bool = True
    diary_enabled: bool = True
    diary_time: str = "23:00"
    diary_timezone: str = "Asia/Shanghai"
    diary_min_turns: int = 4
    diary_prompt: str = ""
    """Custom writing requirements; empty keeps the built-in 2–4 sentence prompt."""
    diary_rewrite_prompt: str = ""
    """Custom rewrite requirements; empty keeps the built-in full-rewrite prompt."""
    diary_digest_max_chars: int = 1800
    diary_tidy_enabled: bool = True
    diary_material_keep_days: int = 3
    memory_curation_enabled: bool = True
    """Run one nightly model pass that merges/updates/archives/adds memories."""
    memory_curation_max_ops: int = 8

    @property
    def has_owner_identity(self) -> bool:
        return bool(self.owner_platform and self.owner_bot_id and self.owner_user_id)

    def is_owner(self, platform: str, bot_id: str, user_id: str) -> bool:
        return self.has_owner_identity and (
            platform, bot_id, user_id
        ) == (self.owner_platform, self.owner_bot_id, self.owner_user_id)

    def owner_alias_for(self, scope: ScopeKey) -> str:
        if scope.kind == "dm" and self.is_owner(scope.platform, scope.bot_id, scope.owner_id):
            return self.owner_alias
        return ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None) -> WarmMemoryConfig:
        """Create a bounded immutable config from AstrBot's plugin config."""
        data = raw or {}
        return cls(
            auto_memory_enabled=_as_bool(data.get("auto_memory_enabled"), True),
            message_dedupe_seconds=_as_float(
                data.get("message_dedupe_seconds"), 10.0, 0.0, 300.0
            ),
            chat_style_guard=_as_bool(data.get("chat_style_guard"), True),
            strip_reasoning_history=_as_bool(data.get("strip_reasoning_history"), True),
            memory_extract_interval=_as_int(data.get("memory_extract_interval"), 12, 1, 200),
            memory_extract_concurrency=_as_int(data.get("memory_extract_concurrency"), 1, 1, 4),
            auto_memory_min_importance=_as_int(data.get("auto_memory_min_importance"), 3, 1, 5),
            memory_capture_ttl_seconds=_as_int(
                data.get("memory_capture_ttl_seconds"), 900, 60, 7200
            ),
            group_memory_enabled=_as_bool(data.get("group_memory_enabled"), False),
            min_saving_ratio=_as_float(data.get("min_saving_ratio"), 0.30, 0.05, 0.90),
            context_window_override=_as_int(data.get("context_window_override"), 0, 0, 2_000_000),
            fallback_context_tokens=_as_int(
                data.get("fallback_context_tokens"), 128000, 512, 2_000_000
            ),
            keep_recent_turns=_as_int(data.get("keep_recent_turns"), 8, 1, 100),
            summary_model_id=_as_string(data.get("summary_model_id")),
            summary_max_chars=_as_int(data.get("summary_max_chars"), 600, 100, 4000),
            memory_recall_limit=_as_int(data.get("memory_recall_limit"), 5, 0, 20),
            memory_recall_max_chars=_as_int(data.get("memory_recall_max_chars"), 800, 0, 5000),
            memory_context_placement=(
                "temporary" if data.get("memory_context_placement") == "temporary" else "system"
            ),
            memory_recall_min_relevance=_as_float(
                data.get("memory_recall_min_relevance"), 0.15, 0.0, 1.0
            ),
            memory_recall_always_include_important=_as_bool(
                data.get("memory_recall_always_include_important"), False
            ),
            memory_recall_always_top_k=_as_int(data.get("memory_recall_always_top_k"), 0, 0, 10),
            memory_recall_fallback=_as_bool(data.get("memory_recall_fallback"), False),
            memory_retrieval_timeout=_as_float(data.get("memory_retrieval_timeout"), 8.0, 0.5, 60.0),
            memory_semantic_enabled=_as_bool(data.get("memory_semantic_enabled"), True),
            memory_semantic_timeout=_as_float(data.get("memory_semantic_timeout"), 6.0, 0.01, 20.0),
            memory_semantic_model_id=_as_string(data.get("memory_semantic_model_id")),
            memory_embedding_enabled=_as_bool(data.get("memory_embedding_enabled"), False),
            memory_embedding_model_id=_as_string(data.get("memory_embedding_model_id")),
            memory_embedding_timeout=_as_float(data.get("memory_embedding_timeout"), 6.0, 0.01, 30.0),
            memory_embedding_min_similarity=_as_float(data.get("memory_embedding_min_similarity"), 0.55, 0.0, 1.0),
            memory_rerank_model_id=_as_string(data.get("memory_rerank_model_id")),
            memory_rerank_timeout=_as_float(data.get("memory_rerank_timeout"), 4.0, 0.01, 20.0),
            memory_rerank_min_score=_as_float(data.get("memory_rerank_min_score"),0.0,0.0,1.0),
            owner_alias=_as_string(data.get("owner_alias")),
            owner_platform=_as_string(data.get("owner_platform")),
            owner_bot_id=_as_string(data.get("owner_bot_id")),
            owner_user_id=_as_string(data.get("owner_user_id")),
            memory_dedupe_context=_as_int(data.get("memory_dedupe_context"), 40, 0, 60),
            base_half_life_days=_as_int(data.get("base_half_life_days"), 30, 1, 3650),
            archive_retention_threshold=_as_float(
                data.get("archive_retention_threshold"), 0.10, 0.001, 0.99
            ),
            destructive_commands_require_confirm=_as_bool(
                data.get("destructive_commands_require_confirm"), True
            ),
            diary_enabled=_as_bool(data.get("diary_enabled"), True),
            diary_time=_as_clock(data.get("diary_time")),
            diary_timezone=_as_string(data.get("diary_timezone"), "Asia/Shanghai")
            or "Asia/Shanghai",
            diary_min_turns=_as_int(data.get("diary_min_turns"), 4, 1, 500),
            diary_prompt=_as_string(data.get("diary_prompt")),
            diary_rewrite_prompt=_as_string(data.get("diary_rewrite_prompt")),
            diary_digest_max_chars=_as_int(data.get("diary_digest_max_chars"), 1800, 200, 20000),
            diary_tidy_enabled=_as_bool(data.get("diary_tidy_enabled"), True),
            diary_material_keep_days=_as_int(data.get("diary_material_keep_days"), 3, 0, 90),
            memory_curation_enabled=_as_bool(data.get("memory_curation_enabled"), True),
            memory_curation_max_ops=_as_int(
                data.get("memory_curation_max_ops"), 8, 1, 20
            ),
        )
