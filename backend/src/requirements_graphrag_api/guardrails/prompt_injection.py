"""Prompt injection detection for user inputs.

Detects prompt injection attacks through pattern matching across five
consolidated categories: instruction override, system prompt extraction,
delimiter injection, role-play attacks, and jailbreak attempts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from requirements_graphrag_api.observability import traceable_safe

if TYPE_CHECKING:
    from collections.abc import Sequence


class InjectionRisk(StrEnum):
    """Risk levels for prompt injection detection."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class InjectionPattern:
    """A pattern for detecting prompt injection attempts."""

    pattern: re.Pattern[str]
    name: str
    description: str
    weight: int = 1


@dataclass(frozen=True, slots=True)
class InjectionCheckResult:
    """Result of a prompt injection check."""

    risk_level: InjectionRisk
    detected_patterns: tuple[str, ...]
    matched_texts: tuple[str, ...]
    total_weight: int
    should_block: bool
    should_warn: bool


# Five consolidated pattern categories (merged from original 21 patterns)
_INJECTION_PATTERNS: tuple[InjectionPattern, ...] = (
    # === Instruction Override (Weight 3) ===
    InjectionPattern(
        pattern=re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)"
            r"|disregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|guidelines?)"
            r"|forget\s+(everything|all|what)\s+(you\s+)?(were|have\s+been)\s+told"
            r"|new\s+(instructions?|rules?|guidelines?):\s*",
            re.IGNORECASE,
        ),
        name="instruction_override",
        description="Attempts to override, disregard, or inject new instructions",
        weight=3,
    ),
    # === System Prompt Extraction (Weight 3) ===
    InjectionPattern(
        pattern=re.compile(
            r"(reveal|show|display|print|output|tell\s+me)\s+(your|the)\s+system\s+prompt"
            r"|what\s+(are|is)\s+(your|the)\s+(initial\s+)?(instructions?|system\s+prompt|rules?)"
            r"|(repeat|echo|recite)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
            re.IGNORECASE,
        ),
        name="system_prompt_extraction",
        description="Attempts to extract system prompt or instructions",
        weight=3,
    ),
    # === Delimiter Injection (Weight 2) ===
    InjectionPattern(
        pattern=re.compile(
            r"```\s*(system|assistant|user|human|ai)[\s\n]"
            r"|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>"
            r"|<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>"
            r"|Human:|Assistant:|System:|HUMAN:|ASSISTANT:|SYSTEM:",
            re.IGNORECASE,
        ),
        name="delimiter_injection",
        description="Chat template or markdown delimiter injection",
        weight=2,
    ),
    # === Role-Play Attack (Weight 2) ===
    InjectionPattern(
        pattern=re.compile(
            r"you\s+are\s+now\s+(a|an|the|in)\s+"
            r"|(pretend|act|behave|respond)\s+(to\s+be|as\s+if\s+you\s+are|like)\s+"
            r"|from\s+now\s+on[,\s]+(you|your|act|respond|answer)",
            re.IGNORECASE,
        ),
        name="role_play_attack",
        description="Attempts to assign a new role or change behavior",
        weight=2,
    ),
    # === Jailbreak Attempt (Weight 3) ===
    InjectionPattern(
        pattern=re.compile(
            r"\bDAN\b.*(?:mode|enabled?|activated?)|Do\s+Anything\s+Now"
            r"|(?:developer|dev|debug|admin|root)\s*(?:mode|access|privileges?)"
            r"|(?:bypass|disable|turn\s+off|ignore)\s+(?:your\s+)?(?:filters?|guardrails?|safety|restrictions?)"
            r"|(?:no\s+)?ethical\s+(?:guidelines?|constraints?|limits?)",
            re.IGNORECASE,
        ),
        name="jailbreak_attempt",
        description="Direct jailbreak or privilege escalation attempt",
        weight=3,
    ),
)


def _calculate_risk_level(total_weight: int, pattern_count: int) -> InjectionRisk:
    """Calculate risk level based on total weight and pattern count."""
    if pattern_count == 0:
        return InjectionRisk.NONE
    if total_weight >= 6 or pattern_count >= 3:
        return InjectionRisk.CRITICAL
    if total_weight >= 4 or pattern_count >= 2:
        return InjectionRisk.HIGH
    if total_weight >= 2:
        return InjectionRisk.MEDIUM
    return InjectionRisk.LOW


@traceable_safe(name="check_prompt_injection", run_type="chain")
def check_prompt_injection(
    text: str,
    block_threshold: InjectionRisk = InjectionRisk.HIGH,
    patterns: Sequence[InjectionPattern] | None = None,
) -> InjectionCheckResult:
    """Check text for prompt injection patterns.

    Args:
        text: The text to check for injection patterns.
        block_threshold: Risk level at or above which to recommend blocking.
        patterns: Optional custom patterns to use instead of defaults.

    Returns:
        InjectionCheckResult with risk assessment and matched patterns.
    """
    patterns_to_check = patterns if patterns is not None else _INJECTION_PATTERNS

    detected: list[str] = []
    matched_texts: list[str] = []
    total_weight = 0

    for injection_pattern in patterns_to_check:
        match = injection_pattern.pattern.search(text)
        if match:
            detected.append(injection_pattern.name)
            matched_texts.append(match.group(0))
            total_weight += injection_pattern.weight

    risk_level = _calculate_risk_level(total_weight, len(detected))

    risk_order = [
        InjectionRisk.NONE,
        InjectionRisk.LOW,
        InjectionRisk.MEDIUM,
        InjectionRisk.HIGH,
        InjectionRisk.CRITICAL,
    ]
    risk_index = risk_order.index(risk_level)
    block_index = risk_order.index(block_threshold)

    should_block = risk_index >= block_index
    should_warn = risk_level not in (InjectionRisk.NONE, InjectionRisk.LOW)

    return InjectionCheckResult(
        risk_level=risk_level,
        detected_patterns=tuple(detected),
        matched_texts=tuple(matched_texts),
        total_weight=total_weight,
        should_block=should_block,
        should_warn=should_warn,
    )
