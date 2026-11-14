"""Translate AstrBot message events into privacy-scoped memory keys."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .domain import ScopeKey

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the core importable alone
    from astrbot.api.event import AstrMessageEvent


def memory_scopes_for_event(
    event: AstrMessageEvent,
    group_memory_enabled: bool,
    explicit_command: bool = False,
) -> tuple[ScopeKey, ...]:
    """Return only the memory scopes this event is permitted to access."""
    bot_id = str(event.get_self_id() or "").strip()
    platform = str(event.get_platform_name() or "").strip()
    user_id = str(event.get_sender_id() or "").strip()
    if not (bot_id and platform and user_id):
        return ()

    if event.is_private_chat():
        return (ScopeKey(bot_id=bot_id, platform=platform, kind="dm", owner_id=user_id),)

    group_id = str(event.get_group_id() or "").strip()
    if not group_memory_enabled or not group_id:
        return ()

    # Commands, mentions, and replies addressed to the bot set this flag in
    # AstrBot's normal event pipeline. Never capture unrelated group traffic.
    if not explicit_command and not bool(getattr(event, "is_at_or_wake_command", False)):
        return ()

    return (
        ScopeKey(bot_id=bot_id, platform=platform, kind="group_shared", group_id=group_id),
        ScopeKey(
            bot_id=bot_id,
            platform=platform,
            kind="group_member",
            group_id=group_id,
            owner_id=user_id,
        ),
    )


def diary_scope_for_event(event: AstrMessageEvent) -> ScopeKey | None:
    """Namespace that owns the diary for this session.

    One diary per private chat and one per group. Unlike memories, diaries do
    not depend on ``group_memory_enabled``: writing a diary is opted into per
    session with ``/日记 开启``, which is a separate decision.
    """
    bot_id = str(event.get_self_id() or "").strip()
    platform = str(event.get_platform_name() or "").strip()
    user_id = str(event.get_sender_id() or "").strip()
    if not (bot_id and platform and user_id):
        return None
    if event.is_private_chat():
        return ScopeKey(bot_id=bot_id, platform=platform, kind="dm", owner_id=user_id)
    group_id = str(event.get_group_id() or "").strip()
    if not group_id:
        return None
    return ScopeKey(bot_id=bot_id, platform=platform, kind="group_shared", group_id=group_id)
