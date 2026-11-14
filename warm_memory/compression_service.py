"""Token estimates and explicitly selected history compression strategies."""

from __future__ import annotations

import asyncio
import json
import math
import unicodedata
from collections.abc import Awaitable, Callable, Sequence

from .config import WarmMemoryConfig
from .domain import (
    CompressionPreview,
    CompressionResult,
    ConversationKey,
    HistoryUnit,
    MemoryRecord,
    ScopeKey,
)
from .memory_service import MemoryService
from .prompts import build_summary_prompt
from .repository import SQLiteRepository

LLMGenerate = Callable[[str, str | None], Awaitable[str]]


def _message_role(message: dict) -> str:
    role = message.get("role")
    return str(role).strip().lower() if isinstance(role, str) else ""


def group_complete_turns(messages: Sequence[dict]) -> list[HistoryUnit]:
    """Group complete user turns with their assistant/tool messages.

    System/developer messages and malformed or incomplete groups are emitted
    as protected units. Compression may remove only units marked complete.
    """
    units: list[HistoryUnit] = []
    current: list[dict] = []
    pending_tools: set[str] = set()
    has_user = False
    has_final_assistant = False
    malformed = False

    def flush() -> None:
        nonlocal current, pending_tools, has_user, has_final_assistant, malformed
        if current:
            complete = has_user and has_final_assistant and not pending_tools and not malformed
            units.append(HistoryUnit(messages=list(current), complete=complete))
        current = []
        pending_tools = set()
        has_user = False
        has_final_assistant = False
        malformed = False

    for message in messages:
        if not isinstance(message, dict):
            flush()
            units.append(
                HistoryUnit(messages=[{"role": "unknown", "content": str(message)}], complete=False)
            )
            continue

        role = _message_role(message)
        if role == "user":
            flush()
            current = [message]
            has_user = True
            continue

        if role in {"system", "developer"}:
            flush()
            units.append(HistoryUnit(messages=[message], complete=False))
            continue

        if role == "assistant":
            if not current or not has_user:
                flush()
                units.append(HistoryUnit(messages=[message], complete=False))
                continue
            current.append(message)
            tool_calls = message.get("tool_calls")
            if tool_calls is not None:
                if not isinstance(tool_calls, list) or has_final_assistant:
                    malformed = True
                    continue
                for tool_call in tool_calls:
                    call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                    if not isinstance(call_id, str) or not call_id:
                        malformed = True
                    else:
                        pending_tools.add(call_id)
                has_final_assistant = False
            elif pending_tools:
                # An assistant continuation before all tool results is not a
                # safe boundary for removing the exchange.
                malformed = True
            else:
                has_final_assistant = True
            continue

        if role == "tool":
            if not current or not has_user:
                flush()
                units.append(HistoryUnit(messages=[message], complete=False))
                continue
            current.append(message)
            tool_id = message.get("tool_call_id")
            if not isinstance(tool_id, str) or tool_id not in pending_tools:
                malformed = True
            else:
                pending_tools.remove(tool_id)
            continue

        # Legacy function results, refusal-only roles, and vendor-specific
        # message shapes are left intact instead of guessed at.
        flush()
        units.append(HistoryUnit(messages=[message], complete=False))

    flush()
    return units


def _remove_old_complete_turns(
    units: Sequence[HistoryUnit], keep_recent_turns: int
) -> tuple[list[dict], list[dict], int]:
    complete_indices = [index for index, unit in enumerate(units) if unit.complete]
    remove_count = max(0, len(complete_indices) - keep_recent_turns)
    remove_indices = set(complete_indices[:remove_count])
    removed: list[dict] = []
    retained: list[dict] = []
    for index, unit in enumerate(units):
        target = removed if index in remove_indices else retained
        target.extend(unit.messages)
    return retained, removed, remove_count


def estimate_tokens(value: object) -> int:
    """Roughly estimate tokens: CJK wide/full-width code point=1, other=0.25."""
    try:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
    except (TypeError, ValueError):
        text = str(value)
    units = 0.0
    for char in text:
        if unicodedata.east_asian_width(char) in {"W", "F"}:
            units += 1.0
        else:
            units += 0.25
    return math.ceil(units)


class CompressionService:
    def __init__(
        self,
        repository: SQLiteRepository,
        config: WarmMemoryConfig,
        llm_generate: LLMGenerate,
    ) -> None:
        self.repository = repository
        self.config = config
        self.llm_generate = llm_generate

    @staticmethod
    def group_complete_turns(messages: list[dict]) -> list[HistoryUnit]:
        return group_complete_turns(messages)

    @staticmethod
    def estimate_tokens(messages: object) -> int:
        return estimate_tokens(messages)

    def _project(
        self,
        messages: list[dict],
        existing_summary: str,
        mode: str,
    ) -> tuple[CompressionPreview, list[dict], list[dict]]:
        units = group_complete_turns(messages)
        retained, removed, removed_turns = _remove_old_complete_turns(
            units,
            self.config.keep_recent_turns,
        )
        before_tokens = estimate_tokens(messages) + estimate_tokens(existing_summary)
        retained_tokens = estimate_tokens(retained)
        old_summary_tokens = estimate_tokens(existing_summary)
        if mode == "truncate":
            after_tokens = retained_tokens + old_summary_tokens
            saving = before_tokens - after_tokens
            preview = CompressionPreview(
                mode=mode,
                removable_turns=removed_turns,
                estimated_before_tokens=before_tokens,
                estimated_after_tokens=after_tokens,
                estimated_net_saving_per_turn=saving,
                notes=("截断不调用摘要模型；被移除的早期细节无法从插件恢复。",),
            )
            return preview, retained, removed

        estimated_summary = estimate_tokens("界" * self.config.summary_max_chars)
        after_tokens = retained_tokens + estimated_summary
        call_input = estimate_tokens(removed) + old_summary_tokens + 160
        call_output = estimate_tokens("界" * (self.config.summary_max_chars + 120))
        net_saving = before_tokens - after_tokens
        break_even = math.ceil((call_input + call_output) / net_saving) if net_saving > 0 else None
        preview = CompressionPreview(
            mode=mode,
            removable_turns=removed_turns,
            estimated_before_tokens=before_tokens,
            estimated_after_tokens=after_tokens,
            estimated_call_input_tokens=call_input,
            estimated_call_output_tokens=call_output,
            estimated_net_saving_per_turn=net_saving,
            break_even_future_turns=break_even,
            notes=("以上为字符启发式估算，不是模型账单；摘要内容可能遗漏早期细节。",),
        )
        return preview, retained, removed

    def preview(
        self,
        messages: list[dict],
        existing_summary: str,
        mode: str = "summary",
    ) -> CompressionPreview:
        if mode not in {"summary", "truncate"}:
            mode = "summary"
        preview, _, _ = self._project(messages, existing_summary, mode)
        return preview

    def split_removable(self, messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return ``(retained, removed)`` using the configured recent-turn budget."""
        retained, removed, _ = self.split_removable_plan(messages)
        return retained, removed

    def split_removable_plan(self, messages: list[dict]) -> tuple[list[dict], list[dict], list[int]]:
        """Identify exact message positions, including gaps for protected units."""
        units = group_complete_turns(messages)
        complete = [index for index, unit in enumerate(units) if unit.complete]
        selected = set(complete[:max(0, len(complete) - self.config.keep_recent_turns)])
        positions: list[int] = []
        offset = 0
        for index, unit in enumerate(units):
            if index in selected:
                positions.extend(range(offset, offset + len(unit.messages)))
            offset += len(unit.messages)
        selected_positions = set(positions)
        retained = [message for index, message in enumerate(messages) if index not in selected_positions]
        removed = [messages[index] for index in positions]
        return retained, removed, positions

    async def compress_truncate(
        self,
        key: ConversationKey,
        messages: list[dict],
        existing_summary: str,
    ) -> CompressionResult:
        preview, retained, removed = self._project(messages, existing_summary, "truncate")
        if not removed:
            return CompressionResult(
                False,
                messages,
                existing_summary,
                preview.estimated_before_tokens,
                preview.estimated_before_tokens,
                reason="没有可安全移除的完整旧轮次。",
                mode="truncate",
            )
        if preview.estimated_net_saving_per_turn <= 0:
            return CompressionResult(
                False,
                messages,
                existing_summary,
                preview.estimated_before_tokens,
                preview.estimated_before_tokens,
                reason="按当前摘要和保留轮数估算，截断不会减少请求 Token。",
                mode="truncate",
            )
        return CompressionResult(
            True,
            retained,
            existing_summary,
            preview.estimated_before_tokens,
            preview.estimated_after_tokens,
            mode="truncate",
        )

    async def compress_summary(
        self,
        key: ConversationKey,
        messages: list[dict],
        existing_summary: str,
        allowed_memory_scopes: Sequence[ScopeKey],
        source_user_id: str,
        provider_id: str | None,
    ) -> CompressionResult:
        preview, retained, removed = self._project(messages, existing_summary, "summary")
        if not removed:
            return CompressionResult(
                False,
                messages,
                existing_summary,
                preview.estimated_before_tokens,
                preview.estimated_before_tokens,
                reason="没有可安全压缩的完整旧轮次。",
                mode="summary",
            )
        existing: list[MemoryRecord] = []
        if self.config.memory_dedupe_context > 0:
            for scope in allowed_memory_scopes:
                existing.extend(
                    await asyncio.to_thread(self.repository.list_memories, scope, False)
                )
            existing.sort(key=lambda record: record.created_at)
            existing = existing[-self.config.memory_dedupe_context :]
        prompt = build_summary_prompt(
            existing_summary,
            removed,
            max_chars=self.config.summary_max_chars,
            allowed_scopes=tuple(scope.kind for scope in allowed_memory_scopes),
            existing=existing,
        )
        raw = await self.llm_generate(prompt, provider_id)
        payload = self._parse_json_object(raw)
        if payload is None:
            return CompressionResult(
                False,
                messages,
                existing_summary,
                preview.estimated_before_tokens,
                preview.estimated_before_tokens,
                reason="摘要模型没有返回有效 JSON，历史未修改。",
                mode="summary",
            )
        summary = payload.get("summary")
        if not isinstance(summary, str):
            return CompressionResult(
                False,
                messages,
                existing_summary,
                preview.estimated_before_tokens,
                preview.estimated_before_tokens,
                reason="摘要模型没有返回摘要文本，历史未修改。",
                mode="summary",
            )
        summary = summary.strip()
        if not summary or len(summary) > self.config.summary_max_chars:
            return CompressionResult(
                False,
                messages,
                existing_summary,
                preview.estimated_before_tokens,
                preview.estimated_before_tokens,
                reason="摘要为空或超过配置长度，历史未修改。",
                mode="summary",
            )

        after_tokens = estimate_tokens(retained) + estimate_tokens(summary)
        savings_ratio = (
            (preview.estimated_before_tokens - after_tokens) / preview.estimated_before_tokens
            if preview.estimated_before_tokens > 0
            else 0.0
        )
        if savings_ratio < self.config.min_saving_ratio:
            return CompressionResult(
                False,
                messages,
                existing_summary,
                preview.estimated_before_tokens,
                preview.estimated_before_tokens,
                reason=f"预计 Token 减少不足 {int(self.config.min_saving_ratio * 100)}%，未应用摘要。",
                mode="summary",
            )

        memory_candidates = MemoryService.parse_candidates(
            raw,
            allowed_memory_scopes,
            source_user_id,
            known_ids=frozenset(record.id for record in existing) or None,
            min_importance=self.config.auto_memory_min_importance,
        )
        return CompressionResult(
            True,
            retained,
            summary,
            preview.estimated_before_tokens,
            after_tokens,
            memory_candidates=memory_candidates,
            mode="summary",
        )

    async def persist(self, key: ConversationKey, result: CompressionResult) -> None:
        """Store the rolling summary only after the new history is in place."""
        await asyncio.to_thread(
            self.repository.save_summary,
            key.bot_id,
            key.platform,
            key.umo,
            key.conversation_id,
            result.summary,
            result.mode or "summary",
            result.estimated_before_tokens,
            result.estimated_after_tokens,
        )

    @staticmethod
    def _parse_json_object(raw: object) -> dict | None:
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
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None
