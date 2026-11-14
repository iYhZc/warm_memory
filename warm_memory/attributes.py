"""Which memories describe a single-valued attribute.

"住址" can only have one current value, so a new wording replaces the old one
even when the two sentences share no characters (``住在杭州`` → ``搬到上海``).
Preferences, hobbies and habits are multi-valued and must never supersede each
other, which is why this is a short allow-list rather than a similarity guess.

The list is deliberately small: 住址 / 学校 / 专业 / 年级 / 生日 / 职业.
"""

from __future__ import annotations

import re

_SINGLE_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("住址", re.compile(r"住在|住址|搬到|搬家|家在|宿舍在|小区|租房")),
    ("学校", re.compile(r"学校|就读|考上|上学|学院|高校|母校|考研|读研|读博")),
    ("专业", re.compile(r"专业|主修")),
    (
        "年级",
        re.compile(r"大一|大二|大三|大四|研一|研二|研三|博一|高一|高二|高三|初一|初二|初三"),
    ),
    ("生日", re.compile(r"生日|出生|星座")),
    ("职业", re.compile(r"职业|工作|上班|就职|岗位|实习|入职|辞职")),
)

ATTRIBUTE_NAMES: tuple[str, ...] = tuple(name for name, _ in _SINGLE_VALUE_PATTERNS)


def attribute_keys(content: str) -> frozenset[str]:
    """All single-valued attributes this memory mentions.

    A sentence like "辞职后去读研" touches both 职业 and 学校; returning the set
    lets the caller detect a conflict on either one.
    """
    text = str(content or "")
    if not text:
        return frozenset()
    return frozenset(name for name, pattern in _SINGLE_VALUE_PATTERNS if pattern.search(text))


def attribute_key(content: str) -> str | None:
    """First single-valued attribute mentioned, for display and testing."""
    keys = attribute_keys(content)
    if not keys:
        return None
    return next(name for name in ATTRIBUTE_NAMES if name in keys)
