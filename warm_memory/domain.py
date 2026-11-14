"""Domain records shared by memory, affinity, storage and compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ScopeKind = Literal["dm", "group_shared", "group_member"]

_SCOPE_LABELS: dict[str, str] = {
    "dm": "私聊",
    "group_shared": "本群",
    "group_member": "该用户",
}


@dataclass(frozen=True, slots=True)
class ScopeKey:
    """Complete privacy boundary for one memory namespace."""

    bot_id: str
    platform: str
    kind: ScopeKind
    group_id: str = ""
    owner_id: str = ""

    def __post_init__(self) -> None:
        if not self.bot_id or not self.platform:
            raise ValueError("bot_id and platform are required for a memory scope")
        if self.kind == "dm" and (self.group_id or not self.owner_id):
            raise ValueError("DM scope requires owner_id and must not have group_id")
        if self.kind == "group_shared" and (not self.group_id or self.owner_id):
            raise ValueError("group-shared scope requires group_id and no owner_id")
        if self.kind == "group_member" and (not self.group_id or not self.owner_id):
            raise ValueError("group-member scope requires group_id and owner_id")

    def storage_tuple(self) -> tuple[str, str, str, str, str]:
        return self.bot_id, self.platform, self.kind, self.group_id, self.owner_id

    def label(self) -> str:
        """Human-readable scope label used when injecting recalled facts."""
        return _SCOPE_LABELS.get(self.kind, self.kind)


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    category: str
    content: str
    importance: int = 3
    confidence: float = 0.8
    replaces_previous: bool = False
    """The extractor believes this fact supersedes an existing similar fact."""
    valid_until: float | None = None
    """Epoch after which the fact is no longer injected; ``None`` is permanent."""
    source_quote: str = ""
    """Verbatim user wording that supports the fact, kept as evidence."""
    duplicate_of: int | None = None
    """Existing memory id this candidate repeats without adding information."""
    replaces_id: int | None = None
    """Existing memory id this candidate supersedes (a fuller or conflicting version)."""


@dataclass(frozen=True, slots=True)
class ScopedMemoryCandidate:
    scope: ScopeKey
    candidate: MemoryCandidate
    source_user_id: str


@dataclass(frozen=True, slots=True)
class CaptureMessage:
    text: str
    source_user_id: str
    row_id: int | None = None
    """Row in ``capture_messages``; lets a claimed batch be acked or released."""
    created_at: float = 0.0
    """Trusted capture time; zero keeps older callers compatible."""


@dataclass(frozen=True, slots=True)
class DiaryTurn:
    """One turn of the day, already shortened to a digest line for the diary."""

    user_text: str
    bot_text: str
    created_at: float = 0.0
    row_id: int | None = None
    """Snapshot high-water mark; only rows read by a write may be consumed."""


@dataclass(frozen=True, slots=True)
class DiaryEntry:
    id: int
    scope: ScopeKey
    seq: int
    """Per-scope 1-based number; the user types it as ``/日记 查看 1``."""
    day: str
    """Local calendar day (``YYYY-MM-DD``) the diary covers."""
    content: str
    feeling: str
    """One short line of the bot's own mood for that day."""
    created_at: float
    source_material: str = ""
    """The exact digest used to write it; lets a later rewrite recover the day."""


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: int
    scope: ScopeKey
    category: str
    content: str
    importance: int
    confidence: float
    source_user_id: str
    created_at: float
    updated_at: float
    last_recalled_at: float | None
    recall_count: int
    mention_count: int
    pinned: bool
    archived: bool
    superseded_by: int | None = None
    """ID of the memory that replaced this one; such rows are archived."""
    valid_until: float | None = None
    """Epoch after which recall skips this row; ``None`` means no expiry."""
    source_quote: str = ""
    """Verbatim user wording the fact was drawn from, shown as evidence."""
    last_mentioned_at: float | None = None
    """Last time the user repeated this fact; refreshes decay like a recall."""




@dataclass(frozen=True, slots=True)
class MemoryOp:
    """One validated memory-list operation produced by the nightly curator."""

    kind: str
    """``update``, ``merge``, ``archive`` or ``add``."""
    ids: tuple[int, ...] = ()
    content: str = ""
    category: str = "other"
    importance: int = 3
    reason: str = ""
    source_quote: str = ""


@dataclass(frozen=True, slots=True)
class ConversationKey:
    bot_id: str
    platform: str
    umo: str
    conversation_id: str


@dataclass(frozen=True, slots=True)
class HistoryUnit:
    messages: list[dict]
    complete: bool


@dataclass(frozen=True, slots=True)
class CompressionPreview:
    mode: str
    removable_turns: int
    estimated_before_tokens: int
    estimated_after_tokens: int
    estimated_call_input_tokens: int = 0
    estimated_call_output_tokens: int = 0
    estimated_net_saving_per_turn: int = 0
    break_even_future_turns: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CompressionResult:
    applied: bool
    messages: list[dict]
    summary: str
    estimated_before_tokens: int
    estimated_after_tokens: int
    memory_candidates: list[ScopedMemoryCandidate] = field(default_factory=list)
    reason: str = ""
    mode: str = ""
