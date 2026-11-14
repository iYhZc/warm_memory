"""Scoped memory capture, decay, recall, and user controls."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import lru_cache

from .attributes import attribute_keys
from .config import WarmMemoryConfig
from .domain import (
    CaptureMessage,
    MemoryCandidate,
    MemoryOp,
    MemoryRecord,
    ScopedMemoryCandidate,
    ScopeKey,
)
from .prompts import build_memory_extraction_prompt
from .repository import SQLiteRepository
from .retrieval import topic_text
from .validity import compute_valid_until, is_expired, local_today, resolve_source_time, resolve_timezone

_ALLOWED_CATEGORIES = {
    "preference",
    "fact",
    "relationship",
    "commitment",
    "project",
    "event",
    "other",
}

# Credentials and identifiers are never worth remembering. The patterns below
# require a credential-like context for the generic word "token" so ordinary
# sentences such as "我的项目用 JWT token 认证" are still storable.
_SENSITIVE_PATTERN = re.compile(
    r"(?i)(密码|口令|验证码|身份证|银行卡|信用卡|api[\s_-]*key|密钥|secret|"
    r"(?:access|api|bearer|auth|refresh|session)[\s_-]*token|"
    r"sk-[a-z0-9_-]{12,}|gh[pousr]_[a-z0-9]{16,}|ak\s*[:=]|password\s*[:=])"
)
_SENSITIVE_PERSONAL_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(住址|家庭地址|家庭住址|详细地址|门牌号|电话号码|手机号码|手机号)"),
    re.compile(r"(病史|患有|确诊|诊断结果|处方药|正在服用|糖尿病|抑郁症|焦虑症)"),
    re.compile(r"(宗教信仰|政治立场|投票倾向|性取向|性生活|民族身份|种族身份)"),
    re.compile(r"(我是.{0,8}(佛教徒|基督徒|天主教徒|穆斯林|犹太教徒|同性恋|双性恋|信佛))"),
    re.compile(
        r"(?i)\b(phone number|home address|medical history|health condition|diagnosed with|"
        r"religious belief|political affiliation|political views|sexual orientation|ethnicity)\b"
    ),
)
_CODE_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# Facts about the assistant, the plugin or the deployment are not user memories.
# "用户已为助手安装了记忆插件" was stored once with confidence 0.95 and then
# injected into every single turn.
_SYSTEM_META_PATTERN = re.compile(
    r"(?i)(?:插件|机器人|助手|智能体|astrbot|提示词|人格卡|系统提示|部署|重载)"
    r".{0,16}?(?:安装|配置|部署|升级|更新|测试|重载|重启|启动|开发)"
    r"|(?:安装|配置|部署|升级|更新|重载|测试).{0,8}?(?:插件|机器人|助手|智能体|astrbot)"
)
# The assistant's own knobs (avatar, voice, persona, plugin settings) are not
# facts about the user. They used to slip through because the prompt alone asked
# the model to skip them; these patterns are the code-side gate.
_ASSISTANT_META_PATTERN = re.compile(
    r"(?i)(?:bot|机器人|助手|智能体|虚拟形象)"
    r".{0,12}?(?:头像|语音|音色|人格卡|人设|昵称|签名|插件|提示词|记忆库|日记功能|记忆功能|语音功能)"
    r".{0,18}?(?:设置|配置|更换|换|改|调|安装|部署|升级|更新|开发|改进|调整|测试|失败|报错|只能|不能|不支持|路径|ENOENT)"
    r"|(?:bot|机器人|助手|智能体|虚拟形象).{0,12}?"
    r"(?:开发|改进|调整|测试|安装|配置|部署|升级|更新|优化|重载|重启)"
)

# Same fact, different wording: "我是一名大学生" and "用户是一名大学生" are one
# fact, and storing both injected the same sentence twice. Containment only
# counts as "same" when almost no content is added ("Python" vs "Python 3"
# is a refinement, not a duplicate, so it still goes through supersede).
_FACT_PRONOUNS = ("对方", "本人", "用户", "我")
_DEDUPE_SIMILARITY = 0.93
_SAME_FACT_CONTAINMENT_RATIO = 0.95
_SUPERSEDE_SIMILARITY = 0.82
_MODEL_REPLACE_SIMILARITY = 0.45

# Exact containment is the strongest signal; partial containment is ranked just
# below it so that a long fact containing the whole query beats a short fact
# that happens to share a couple of characters.
_CONTAINMENT_WEIGHT = 0.9
_MIN_RELEVANT_QUERY_CHARS = 2


def _normal_form(text: str) -> str:
    lowered = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in lowered if ch.isalnum())


def _fact_form(text: str) -> str:
    """Normalise away the pronoun so "我" and "用户" describe the same person."""
    normalized = _normal_form(text)
    for pronoun in _FACT_PRONOUNS:
        normalized = normalized.replace(pronoun, "")
    return normalized


def _fact_grams(text: str) -> frozenset[str]:
    form = _fact_form(text) or _normal_form(text)
    return _bigrams_of_normalized(form)


@lru_cache(maxsize=8192)
def _bigrams_of_normalized(normalized: str) -> frozenset[str]:
    if len(normalized) < 2:
        return frozenset({normalized}) if normalized else frozenset()
    return frozenset(normalized[index : index + 2] for index in range(len(normalized) - 1))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _same_fact(left: str, right: str) -> bool:
    """True when two phrasings describe the same fact after pronoun removal."""
    first = _fact_form(left)
    second = _fact_form(right)
    if not first or not second:
        return bool(_normal_form(left)) and _normal_form(left) == _normal_form(right)
    if first == second:
        return True
    shorter, longer = (first, second) if len(first) <= len(second) else (second, first)
    if shorter in longer and len(shorter) >= _SAME_FACT_CONTAINMENT_RATIO * len(longer):
        return True
    return _jaccard(_bigrams_of_normalized(first), _bigrams_of_normalized(second)) >= (
        _DEDUPE_SIMILARITY
    )


def _strip_json_fence(raw: str) -> str:
    return _CODE_FENCE_PATTERN.sub("", raw.strip()).strip()


def _is_sensitive(text: str) -> bool:
    return bool(_SENSITIVE_PATTERN.search(text)) or any(
        pattern.search(text) for pattern in _SENSITIVE_PERSONAL_PATTERNS
    )


def _is_system_meta(text: str) -> bool:
    return bool(_SYSTEM_META_PATTERN.search(text)) or bool(_ASSISTANT_META_PATTERN.search(text))


def is_safe_memory_text(text: str) -> bool:
    """True when text is short enough and free of secrets / assistant meta."""
    value = str(text or "").strip()
    return bool(value) and len(value) <= 240 and not _is_sensitive(value) and not _is_system_meta(value)


def _optional_memory_id(value: object, known_ids: frozenset[int] | None) -> int | None:
    """Accept only a positive int that refers to a memory shown to the model."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    if known_ids is not None and value not in known_ids:
        return None
    return value


def _resolve_quote(
    raw_quote: object,
    source_index: object,
    source_texts: Mapping[int, str] | None,
) -> str:
    """Keep only a verbatim slice of the user's own message as evidence.

    A model-written "quote" that does not literally occur in the cited message
    is discarded in favour of the message itself, so evidence cannot be
    hallucinated.
    """
    quote = str(raw_quote or "").strip()
    source = ""
    if isinstance(source_index, int) and source_texts is not None:
        source = str(source_texts.get(source_index) or "").strip()
    if quote and source and quote in source:
        return quote[:200]
    if source:
        return source[:200]
    return quote[:200]


def render_memory_lines(records: Sequence[MemoryRecord], *, max_chars: int) -> str:
    """Render recalled facts as remembered sentences, not as a record list.

    Bullets, dates and a leading "用户" made the block read like a dossier the
    model answered in report voice.  Facts are joined into one short paragraph;
    the extractor stores them without a "用户" subject, so the older rows get a
    light prefix cleanup here as well.
    """
    parts: list[str] = []
    used = 0
    for record in records:
        content = re.sub(r"^(?:该)?用户(?=[\u4e00-\u9fff])", "", record.content.strip())
        content = content.rstrip("。；;，,")
        if not content:
            continue
        prefix = "本群：" if record.scope.kind == "group_shared" else ""
        line = f"{prefix}{content}"
        if used + len(line) > max_chars:
            continue
        parts.append(line)
        used += len(line) + 1
    if not parts:
        return ""
    return "；".join(parts) + "。"


class MemoryCaptureBuffer:
    """Durable batch queue for user messages awaiting extraction.

    Rows live in SQLite (``capture_messages``), so a plugin reload no longer
    loses the turns that had not reached a full batch yet. A batch is claimed
    (state ``inflight``) before the model call and only deleted on success: a
    crash or a failed call re-offers the same messages instead of duplicating
    or dropping them.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        batch_size: int,
        *,
        ttl_seconds: float = 900.0,
        max_attempts: int = 3,
    ) -> None:
        self.repository = repository
        self.batch_size = max(1, int(batch_size))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_attempts = max(1, int(max_attempts))

    async def stage(
        self,
        event_id: str,
        scopes: Sequence[ScopeKey],
        user_text: str,
        source_user_id: str,
        now: float | None = None,
    ) -> bool:
        """Persist one user message for every scope it belongs to."""
        key = event_id.strip()
        text = user_text.strip()
        if not key or not text or not scopes:
            return False
        timestamp = time.time() if now is None else now
        await asyncio.to_thread(self.repository.prune_pending_capture, self.ttl_seconds, timestamp)
        inserted = await asyncio.to_thread(
            self.repository.save_capture,
            key,
            tuple(scopes),
            text,
            source_user_id,
            timestamp,
        )
        return inserted > 0

    async def finish(
        self,
        event_id: str,
        *,
        success: bool,
        now: float | None = None,
    ) -> dict[ScopeKey, tuple[CaptureMessage, ...]]:
        """Keep the staged message only for a successful turn, then take a batch."""
        key = event_id.strip()
        if not key:
            return {}
        if not success:
            await asyncio.to_thread(self.repository.delete_capture, key)
            return {}
        await asyncio.to_thread(self.repository.mark_capture_buffered, key)
        return await self.take_due()

    async def discard(self, event_id: str) -> None:
        await asyncio.to_thread(self.repository.delete_capture, event_id.strip())

    async def take_due(self) -> dict[ScopeKey, tuple[CaptureMessage, ...]]:
        return await asyncio.to_thread(self.repository.claim_capture_batch, self.batch_size)

    async def acknowledge(self, batch: Sequence[CaptureMessage]) -> None:
        """Remove an extracted batch for good."""
        row_ids = [message.row_id for message in batch if message.row_id is not None]
        if row_ids:
            await asyncio.to_thread(self.repository.delete_capture_ids, row_ids)

    async def release(self, batch: Sequence[CaptureMessage], *, failed_attempt: bool = True) -> bool:
        """Return a failed batch to the queue; False when retries are exhausted."""
        row_ids = [message.row_id for message in batch if message.row_id is not None]
        if not row_ids:
            return False
        dropped = await asyncio.to_thread(
            self.repository.release_capture_ids, row_ids, self.max_attempts,
            failed_attempt=failed_attempt,
        )
        return dropped == 0

    async def pending_count(self) -> int:
        return await asyncio.to_thread(self.repository.count_capture, "pending")

    async def buffered_count(self) -> int:
        return await asyncio.to_thread(self.repository.count_capture, "buffered")


class MemoryService:
    def __init__(self, repository: SQLiteRepository, config: WarmMemoryConfig) -> None:
        self.repository = repository
        self.config = config

    @staticmethod
    def retention(record: MemoryRecord, now: float, base_half_life_days: int) -> float:
        if record.pinned:
            return 1.0
        effective_half_life = (
            base_half_life_days
            * (0.5 + record.importance / 5)
            * (
                1
                + 0.15 * min(max(record.mention_count - 1, 0), 4)
                + 0.25 * min(record.recall_count, 3)
            )
        )
        reference = max(
            record.created_at,
            record.last_recalled_at or 0.0,
            record.last_mentioned_at or 0.0,
        )
        age_days = max(0.0, now - reference) / 86400
        return 0.5 ** (age_days / effective_half_life)

    @staticmethod
    def relevance(query: str, content: str) -> float:
        """Return 0..1 relevance that does not collapse on long memories.

        Jaccard divides by the union, so a short query matched verbatim inside a
        long fact scores near zero (measured 0.09 for an exact "冰美式" hit).
        Exact containment now scores 1.0 and partial containment just below it;
        single-character queries stay at 0 because they match almost anything.
        """
        normalized_query = _normal_form(topic_text(query))
        normalized_content = _normal_form(topic_text(content))
        if len(normalized_query) < _MIN_RELEVANT_QUERY_CHARS or not normalized_content:
            return 0.0
        if normalized_query in normalized_content:
            return 1.0

        query_grams = _bigrams_of_normalized(normalized_query)
        content_grams = _bigrams_of_normalized(normalized_content)
        grams_jaccard = _jaccard(query_grams, content_grams)
        grams_contained = (
            len(query_grams & content_grams) / len(query_grams) if query_grams else 0.0
        )

        query_words = frozenset(re.findall(r"[a-z0-9]+", normalized_query))
        content_words = frozenset(re.findall(r"[a-z0-9]+", normalized_content))
        word_jaccard = _jaccard(query_words, content_words)
        word_contained = len(query_words & content_words) / len(query_words) if query_words else 0.0
        return max(
            grams_jaccard,
            grams_contained * _CONTAINMENT_WEIGHT,
            word_jaccard,
            word_contained * _CONTAINMENT_WEIGHT,
        )

    def _score(self, record: MemoryRecord, relevance: float, retention: float) -> float:
        score = (
            0.45 * relevance
            + 0.25 * (record.importance / 5)
            + 0.15 * record.confidence
            + 0.15 * retention
        )
        if record.pinned:
            score += 0.20
        return score

    def _keeps(self, record: MemoryRecord, relevance: float, retention: float) -> bool:
        """Select facts worth injecting even without lexical overlap.

        Character-bigram similarity is close to zero for most (query, fact)
        pairs in Chinese, so a hard relevance threshold alone hides almost every
        stored fact. Pinned facts and important, still-fresh facts stay eligible.
        """
        if relevance >= self.config.memory_recall_min_relevance:
            return True
        if record.pinned:
            return True
        return (
            self.config.memory_recall_always_include_important
            and record.importance >= 4
            and retention >= 0.25
        )

    async def recall(
        self,
        scopes: Sequence[ScopeKey],
        query: str,
        now: float | None = None,
        *,
        semantic_ids: Sequence[int] = (),
        semantic_scores: Mapping[int, float] | None = None,
        expected_content: Mapping[int, str] | None = None,
        ranked_ids: Sequence[int] | None = None,
    ) -> list[MemoryRecord]:
        timestamp = time.time() if now is None else now
        qualified: list[tuple[float, MemoryRecord]] = []
        fallback: list[tuple[float, MemoryRecord]] = []
        pool: list[MemoryRecord] = []
        retentions: dict[int, float] = {}
        for scope in scopes:
            records = await asyncio.to_thread(self.repository.list_memories, scope, False)
            expired: list[int] = []
            for record in records:
                if is_expired(record.valid_until, timestamp):
                    # Past its window: keep the row for the record, never inject.
                    continue
                retention = self.retention(record, timestamp, self.config.base_half_life_days)
                if retention < self.config.archive_retention_threshold and not record.pinned:
                    expired.append(record.id)
                    continue
                retentions[record.id] = retention
                pool.append(record)
                relevance = self.relevance(query, record.content)
                version_matches = expected_content is None or expected_content.get(record.id) == record.content
                semantic_match = version_matches and (record.id in semantic_ids or bool(semantic_scores and record.id in semantic_scores))
                if version_matches:
                    if record.id in semantic_ids: relevance = 1.0
                    if semantic_scores and record.id in semantic_scores:
                        relevance = max(relevance, semantic_scores[record.id])
                scored = (self._score(record, relevance, retention), record)
                if semantic_match or self._keeps(record, relevance, retention):
                    qualified.append(scored)
                else:
                    fallback.append(scored)
            if expired:
                await asyncio.to_thread(self.repository.archive_memories, scope, expired, timestamp)

        if not qualified and fallback and self.config.memory_recall_fallback:
            fallback.sort(key=lambda item: item[0], reverse=True)
            qualified = fallback[:1]

        top_k = self.config.memory_recall_always_top_k
        if top_k > 0 and pool:
            already = {record.id for _, record in qualified}
            guaranteed = sorted(
                (record for record in pool if record.id not in already),
                key=lambda record: (
                    record.pinned,
                    record.importance,
                    retentions.get(record.id, 0.0),
                ),
                reverse=True,
            )[:top_k]
            for record in guaranteed:
                relevance = self.relevance(query, record.content)
                qualified.append(
                    (self._score(record, relevance, retentions.get(record.id, 0.0)), record)
                )

        qualified.sort(key=lambda item: item[0], reverse=True)
        if ranked_ids is not None:
            order = {ident:index for index,ident in enumerate(ranked_ids)}
            qualified = [(score,r) for score,r in qualified if r.pinned or (
                r.id in order and (expected_content is None or expected_content.get(r.id) == r.content))]
            qualified.sort(key=lambda item:(not item[1].pinned,order.get(item[1].id,-1),-item[0]))
        chosen: list[MemoryRecord] = []
        used_chars = 0
        for _, record in qualified:
            if len(chosen) >= self.config.memory_recall_limit:
                break
            next_chars = len(record.content) + 1
            if used_chars + next_chars > self.config.memory_recall_max_chars:
                continue
            current = await asyncio.to_thread(self.repository.get_memory, record.scope, record.id)
            if current is None or current.archived or current.content != record.content or is_expired(current.valid_until,time.time() if now is None else now):
                continue
            chosen.append(record)
            used_chars += next_chars

        by_scope: dict[ScopeKey, list[int]] = {}
        for record in chosen:
            by_scope.setdefault(record.scope, []).append(record.id)
        for scope, memory_ids in by_scope.items():
            await asyncio.to_thread(self.repository.record_recalls, scope, memory_ids, timestamp)
        return chosen

    async def recall_candidates(self, scopes: Sequence[ScopeKey], query: str) -> list[MemoryRecord]:
        """Read-only eligible pool; never revive archived or expired records."""
        now = time.time()
        records: dict[int, MemoryRecord] = {}
        for scope in scopes:
            for record in await asyncio.to_thread(self.repository.list_memories, scope, False):
                if is_expired(record.valid_until, now):
                    continue
                if self.retention(record, now, self.config.base_half_life_days) < self.config.archive_retention_threshold:
                    continue
                records[record.id] = record
        ranked = sorted(records.values(), key=lambda r: (-self.relevance(query, r.content), -r.importance, r.id))
        chosen: list[MemoryRecord] = []
        chars = 0
        for record in ranked:
            if len(chosen) >= 80:
                break
            if chars + len(record.content) + 40 > 16000:
                continue
            chosen.append(record)
            chars += len(record.content) + 40
        return sorted(chosen, key=lambda r: r.id)

    async def search(
        self, scope: ScopeKey, query: str, limit: int = 10
    ) -> list[tuple[MemoryRecord, float]]:
        """Rank stored facts against a free-text query for ``/记忆 搜索``."""
        text = query.strip()
        if not text:
            return []
        records = await asyncio.to_thread(self.repository.list_memories, scope, True)
        hits: list[tuple[MemoryRecord, float]] = []
        for record in records:
            score = self.relevance(text, record.content)
            if score <= 0.0:
                continue
            hits.append((record, score))
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[: max(1, limit)]

    async def remember(
        self,
        scope: ScopeKey,
        text: str,
        source_user_id: str,
        importance: int = 3,
    ) -> MemoryRecord:
        content = text.strip()
        if not content:
            raise ValueError("记忆内容不能为空")
        if len(content) > 240:
            raise ValueError("单条记忆不能超过 240 个字符")
        if _is_sensitive(content):
            raise ValueError("出于安全考虑，不保存明显的密码、密钥、验证码或联系方式。")
        candidate = MemoryCandidate(
            category="other",
            content=content,
            importance=min(max(int(importance), 1), 5),
            confidence=1.0,
            valid_until=compute_valid_until(
                content,
                None,
                now=time.time(),
                timezone=self.config.diary_timezone,
            ),
        )
        return await asyncio.to_thread(
            self.repository.insert_memory,
            scope,
            candidate,
            source_user_id,
        )

    async def correct(self, scope: ScopeKey, memory_id: int, content: str) -> bool:
        normalized = content.strip()
        if not normalized or len(normalized) > 240:
            raise ValueError("更正内容需为 1–240 个字符")
        if _is_sensitive(normalized):
            raise ValueError("出于安全考虑，不保存明显的密码、密钥、验证码或联系方式。")
        return await asyncio.to_thread(
            self.repository.update_memory_content, scope, memory_id, normalized
        )

    async def pin(self, scope: ScopeKey, memory_id: int, pinned: bool) -> bool:
        updated = await asyncio.to_thread(
            self.repository.set_memory_pinned, scope, memory_id, pinned
        )
        if pinned and updated:
            await asyncio.to_thread(self.repository.set_memory_archived, scope, memory_id, False)
        return updated

    async def forget(self, scope: ScopeKey, memory_id: int) -> bool:
        return await asyncio.to_thread(self.repository.delete_memory, scope, memory_id)

    async def clear(self, scope: ScopeKey) -> int:
        return await asyncio.to_thread(self.repository.clear_scope, scope)

    async def list_memories(
        self, scope: ScopeKey, include_archived: bool = False
    ) -> list[MemoryRecord]:
        return await asyncio.to_thread(self.repository.list_memories, scope, include_archived)

    async def list_archived(self, scope: ScopeKey) -> list[MemoryRecord]:
        records = await asyncio.to_thread(self.repository.list_memories, scope, True)
        return [record for record in records if record.archived]

    async def restore(self, scope: ScopeKey, memory_id: int) -> bool:
        return await asyncio.to_thread(self.repository.restore_memory, scope, memory_id)

    async def maintain(self, retain_days: int = 90) -> dict[str, int]:
        """Drop long-retired rows, expired captures, and compact the database."""
        removed = await asyncio.to_thread(self.repository.prune_superseded, retain_days)
        pruned_captures = await asyncio.to_thread(self.repository.prune_all_pending_capture)
        await asyncio.to_thread(self.repository.vacuum)
        return {
            "superseded_removed": removed,
            "pending_captures_removed": pruned_captures,
        }

    async def apply_curation(
        self,
        scope: ScopeKey,
        ops: Sequence[MemoryOp],
        *,
        source_user_id: str = "",
    ) -> list[str]:
        """Apply validated nightly ``update``/``merge``/``archive``/``add`` ops.

        Pinned memories are never modified automatically, and nothing is hard
        deleted: an obsolete row is archived, so it stays recoverable with
        ``/记忆 恢复``.
        """
        if not ops:
            return []
        records = await self.list_memories(scope, include_archived=False)
        by_id = {record.id: record for record in records}
        actions: list[str] = []
        now = time.time()
        for op in ops:
            if op.kind == "add":
                content = re.sub(
                    r"^(?:该)?用户(?=[\u4e00-\u9fff])", "", op.content.strip()
                )
                if not is_safe_memory_text(content):
                    continue
                if scope.kind == "group_shared" and not source_user_id:
                    continue
                candidate = MemoryCandidate(
                    category=op.category if op.category in _ALLOWED_CATEGORIES else "other",
                    content=content,
                    importance=min(max(int(op.importance), 1), 5),
                    confidence=0.8,
                    source_quote=op.source_quote,
                )
                merged = await self.merge_candidates(
                    scope, [candidate], source_user_id or scope.owner_id
                )
                if merged:
                    actions.append(f"新增「{merged[0].content}」")
                continue

            ids = [memory_id for memory_id in op.ids if memory_id in by_id]
            target = by_id.get(ids[0]) if ids else None
            if target is None:
                continue

            if op.kind == "archive":
                if target.pinned:
                    continue
                archived = await asyncio.to_thread(
                    self.repository.set_memory_archived, scope, target.id, True, now
                )
                if archived:
                    actions.append(f"归档「{target.content[:16]}」")
                continue

            if op.kind not in {"update", "merge"}:
                continue
            if target.pinned:
                continue
            content = re.sub(
                r"^(?:该)?用户(?=[\u4e00-\u9fff])", "", op.content.strip()
            )
            if not is_safe_memory_text(content):
                continue
            valid_until = compute_valid_until(
                content, None, now=now, timezone=self.config.diary_timezone
            )
            changed = await asyncio.to_thread(
                self.repository.update_memory, scope, target.id, content, valid_until, now,
                source_quote=op.source_quote or None,
            )
            if changed:
                actions.append(f"更新「{target.content[:16]}」→「{content[:16]}」")
            if op.kind == "merge":
                for other_id in ids[1:]:
                    other = by_id.get(other_id)
                    if other is None or other.pinned:
                        continue
                    await asyncio.to_thread(
                        self.repository.supersede_memory, scope, other.id, target.id, now
                    )
                    if other.mention_count:
                        await asyncio.to_thread(
                            self.repository.add_memory_mentions,
                            scope,
                            target.id,
                            other.mention_count,
                            now,
                        )
                    actions.append(f"并入「{other.content[:16]}」")
        return actions

    @staticmethod
    def parse_candidates(
        raw: str,
        allowed_scopes: Sequence[ScopeKey],
        source_user_id: str,
        *,
        source_user_ids: Mapping[int, str] | None = None,
        source_texts: Mapping[int, str] | None = None,
        source_times: Mapping[int, float] | None = None,
        known_ids: frozenset[int] | None = None,
        min_importance: int = 1,
        now: float | None = None,
        timezone: str = "Asia/Shanghai",
        strict: bool = False,
    ) -> list[ScopedMemoryCandidate]:
        timestamp = time.time() if now is None else now
        text = _strip_json_fence(raw)
        if not text:
            if strict:
                raise ValueError("记忆提取响应为空")
            return []
        if not text.startswith("{"):
            left, right = text.find("{"), text.rfind("}")
            if left < 0 or right <= left:
                if strict:
                    raise ValueError("记忆提取响应不是JSON对象")
                return []
            text = text[left : right + 1]
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            if strict:
                raise ValueError("记忆提取响应JSON格式无效") from None
            return []
        if not isinstance(payload, Mapping):
            if strict:
                raise ValueError("记忆提取响应不是JSON对象")
            return []
        raw_items = payload.get("memories")
        if not isinstance(raw_items, list):
            if strict:
                raise ValueError("记忆提取响应缺少memories列表")
            return []

        threshold = min(max(int(min_importance), 1), 5)
        scopes_by_name: dict[str, ScopeKey] = {}
        for scope in allowed_scopes:
            scopes_by_name[scope.kind] = scope
            if scope.kind == "group_shared":
                scopes_by_name["shared"] = scope
            elif scope.kind == "group_member":
                scopes_by_name["member"] = scope
        parsed: list[ScopedMemoryCandidate] = []
        for item in raw_items[:12]:
            if not isinstance(item, Mapping):
                continue
            target_scope = scopes_by_name.get(str(item.get("scope", "")).strip().lower())
            category = str(item.get("category", "other")).strip().lower()
            content = str(item.get("content", "")).strip()
            if (
                target_scope is None
                or not content
                or len(content) > 240
                or _is_sensitive(content)
                or _is_system_meta(content)
            ):
                continue
            try:
                importance = min(max(int(item.get("importance", 3)), 1), 5)
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError, OverflowError):
                continue
            if importance < threshold:
                continue
            if not math.isfinite(confidence) or confidence < 0.5:
                continue
            candidate = MemoryCandidate(
                category=category if category in _ALLOWED_CATEGORIES else "other",
                content=content,
                importance=importance,
                confidence=min(confidence, 1.0),
                replaces_previous=bool(item.get("replaces_previous", False)),
                valid_until=compute_valid_until(
                    content,
                    item.get("valid_until"),
                    now=resolve_source_time(
                        source_times.get(item.get("source_index"))
                        if source_times is not None and type(item.get("source_index")) is int else None,
                        timestamp,
                    ),
                    timezone=timezone,
                ),
                source_quote=_resolve_quote(
                    item.get("source_quote"), item.get("source_index"), source_texts
                ),
                duplicate_of=_optional_memory_id(item.get("duplicate_of"), known_ids),
                replaces_id=_optional_memory_id(item.get("replaces_id"), known_ids),
            )
            candidate_source = source_user_id
            source_index = item.get("source_index")
            resolved_source: str | None = None
            if source_user_ids is not None and isinstance(source_index, int):
                resolved_source = source_user_ids.get(source_index)
            if target_scope.kind == "group_shared":
                # A fact shared with the whole group must stay traceable to the
                # member who said it; without that provenance it is dropped.
                if resolved_source is None:
                    continue
                candidate_source = resolved_source
            elif resolved_source is not None:
                candidate_source = resolved_source
            parsed.append(ScopedMemoryCandidate(target_scope, candidate, candidate_source))
        return parsed

    @staticmethod
    def _best_match(
        existing: Sequence[MemoryRecord],
        content: str,
    ) -> tuple[MemoryRecord | None, float]:
        grams = _fact_grams(content)
        best: MemoryRecord | None = None
        best_score = 0.0
        for record in existing:
            score = _jaccard(grams, _fact_grams(record.content))
            if score > best_score:
                best = record
                best_score = score
        return best, best_score

    async def merge_candidates(
        self,
        scope: ScopeKey,
        candidates: Sequence[MemoryCandidate],
        source_user_id: str,
    ) -> list[MemoryRecord]:
        all_records = await asyncio.to_thread(self.repository.list_memories, scope, True)
        existing = [record for record in all_records if record.superseded_by is None]
        by_id = {record.id: record for record in existing}
        merged: list[MemoryRecord] = []
        now = time.time()
        for candidate in candidates:
            content = candidate.content.strip()
            if not content or len(content) > 240:
                continue
            if _is_sensitive(content) or _is_system_meta(content):
                continue
            safe = MemoryCandidate(
                category=candidate.category
                if candidate.category in _ALLOWED_CATEGORIES
                else "other",
                content=content,
                importance=min(max(candidate.importance, 1), 5),
                confidence=min(max(candidate.confidence, 0.0), 1.0),
                replaces_previous=candidate.replaces_previous,
                valid_until=candidate.valid_until,
                source_quote=(candidate.source_quote or "")[:200],
                duplicate_of=candidate.duplicate_of,
                replaces_id=candidate.replaces_id,
            )
            # The model saw this scope's memories, so an explicit "same fact" is
            # trusted over lexical overlap: bump the existing row and stop.
            duplicate = by_id.get(safe.duplicate_of) if safe.duplicate_of is not None else None
            if duplicate is not None:
                await self._touch_memory(scope, duplicate, now, safe.valid_until)
                refreshed = await asyncio.to_thread(self.repository.get_memory, scope, duplicate.id)
                if refreshed is not None:
                    merged.append(refreshed)
                continue

            match, similarity = self._best_match(existing, safe.content)
            if match is not None and _same_fact(match.content, safe.content):
                await self._touch_memory(scope, match, now, safe.valid_until)
                refreshed = await asyncio.to_thread(self.repository.get_memory, scope, match.id)
                if refreshed is not None:
                    merged.append(refreshed)
                continue

            attribute = attribute_keys(content)
            clash: MemoryRecord | None = None
            if attribute:
                clash = next(
                    (
                        item
                        for item in existing
                        if not item.archived
                        and attribute_keys(item.content) & attribute
                    ),
                    None,
                )

            replaced_match = match
            replaced = match is not None and (
                similarity >= _SUPERSEDE_SIMILARITY
                or (safe.replaces_previous and similarity >= _MODEL_REPLACE_SIMILARITY)
            )
            explicit = by_id.get(safe.replaces_id) if safe.replaces_id is not None else None
            if explicit is not None:
                # "This is a fuller/changed version of [id]" from the extractor.
                replaced = True
                replaced_match = explicit
            if not replaced and clash is not None:
                # 住址/学校/专业/年级/生日/职业 can only hold one current value,
                # so a new wording retires the old row even without overlap.
                replaced = True
                replaced_match = clash
            record = await asyncio.to_thread(
                self.repository.insert_memory, scope, safe, source_user_id, now
            )
            if replaced and replaced_match is not None:
                await asyncio.to_thread(
                    self.repository.supersede_memory, scope, replaced_match.id, record.id, now
                )
                await self._inherit_memory(scope, replaced_match, record.id, now)
                by_id.pop(replaced_match.id, None)
                existing = [item for item in existing if item.id != replaced_match.id]
            existing.append(record)
            by_id[record.id] = record
            merged.append(record)
        return merged

    async def _touch_memory(
        self,
        scope: ScopeKey,
        record: MemoryRecord,
        now: float,
        valid_until: float | None = None,
    ) -> None:
        """Count one more mention and refresh its decay clock.

        Repeating a fact is evidence that it is still current, so a formerly
        archived row is revived and its age restarts from this mention. A later
        validity window from the new wording is inherited as well.
        """
        await asyncio.to_thread(
            self.repository.reinforce_memory,
            scope,
            record.id,
            now,
            valid_until,
        )

    async def _inherit_memory(
        self,
        scope: ScopeKey,
        old: MemoryRecord,
        new_id: int,
        now: float,
    ) -> None:
        """Carry the old row's repetition and pin signals onto its replacement."""
        if old.mention_count > 0:
            await asyncio.to_thread(
                self.repository.add_memory_mentions, scope, new_id, old.mention_count, now
            )
        if old.pinned:
            await asyncio.to_thread(self.repository.set_memory_pinned, scope, new_id, True, now)

    async def _existing_for_prompt(self, scopes: Sequence[ScopeKey]) -> list[MemoryRecord]:
        """Newest active memories of these scopes, shown to the extractor.

        The list is what lets the model answer "same fact" or "fuller version"
        with an explicit id instead of relying on character overlap.
        """
        limit = self.config.memory_dedupe_context
        if limit <= 0:
            return []
        records: list[MemoryRecord] = []
        for scope in scopes:
            records.extend(await asyncio.to_thread(self.repository.list_memories, scope, False))
        records.sort(key=lambda record: record.created_at)
        return records[-limit:]

    async def extract_and_merge(
        self,
        user_messages: Sequence[CaptureMessage | str],
        allowed_scopes: Sequence[ScopeKey],
        source_user_id: str,
        generate: Callable[[str], Awaitable[str]],
    ) -> list[MemoryRecord]:
        """Use one batched model call, then persist only validated candidates."""
        if not user_messages or not allowed_scopes:
            return []
        timestamp = time.time()
        today = local_today(timestamp, resolve_timezone(self.config.diary_timezone)).isoformat()
        group_mode = any(scope.kind.startswith("group_") for scope in allowed_scopes)
        existing = await self._existing_for_prompt(allowed_scopes)
        prompt = build_memory_extraction_prompt(
            user_messages,
            group_mode=group_mode,
            today=today,
            existing=existing,
            owner_alias="" if group_mode else self.config.owner_alias_for(allowed_scopes[0]),
            timezone=self.config.diary_timezone,
        )
        raw = await generate(prompt)
        source_user_ids = {
            index: message.source_user_id
            for index, message in enumerate(user_messages)
            if isinstance(message, CaptureMessage)
        }
        source_texts = {
            index: (message.text if isinstance(message, CaptureMessage) else str(message))
            for index, message in enumerate(user_messages)
        }
        source_times = {
            index: message.created_at
            for index, message in enumerate(user_messages)
            if isinstance(message, CaptureMessage)
        }
        scoped_candidates = self.parse_candidates(
            raw,
            allowed_scopes,
            source_user_id,
            source_user_ids=source_user_ids or None,
            source_texts=source_texts,
            source_times=source_times,
            known_ids=frozenset(record.id for record in existing) or None,
            min_importance=self.config.auto_memory_min_importance,
            now=timestamp,
            timezone=self.config.diary_timezone,
            strict=True,
        )
        grouped: dict[tuple[ScopeKey, str], list[MemoryCandidate]] = {}
        for item in scoped_candidates:
            group_key = (item.scope, item.source_user_id)
            grouped.setdefault(group_key, []).append(item.candidate)
        merged: list[MemoryRecord] = []
        for (scope, source_user_id), candidates in grouped.items():
            merged.extend(await self.merge_candidates(scope, candidates, source_user_id))
        return merged

    async def tidy_duplicates(self, scopes: Sequence[ScopeKey]) -> int:
        """Merge near-identical active memories, keeping one original wording.

        The diary runs this nightly. It never rewrites text and never calls the
        model: only phrasings that describe the same fact after pronoun removal
        are folded together, the oldest wording survives, and pinned entries are
        left alone (two pinned duplicates stay visible on purpose). Retired
        copies keep their text and get ``superseded_by`` so nothing is lost and
        no duplicate is rendered in prompts or ``/记忆 列表`` again.
        """
        merged = 0
        for scope in scopes:
            records = await asyncio.to_thread(self.repository.list_memories, scope, False)
            survivors: list[MemoryRecord] = []
            for record in sorted(records, key=lambda item: (not item.pinned, item.id)):
                match = next(
                    (
                        survivor
                        for survivor in survivors
                        if survivor.pinned == record.pinned
                        and _same_fact(survivor.content, record.content)
                    ),
                    None,
                )
                if match is None:
                    survivors.append(record)
                    continue
                if record.pinned:
                    continue
                await asyncio.to_thread(
                    self.repository.supersede_memory, scope, record.id, match.id
                )
                await asyncio.to_thread(self.repository.increment_memory_mentions, scope, match.id)
                merged += 1
        return merged
