"""Prompt injection detection for user inputs.

This module provides pattern-based detection for prompt injection attacks,
which attempt to manipulate LLM behavior through crafted inputs.

Detection Categories:
    - Instruction override: "ignore previous instructions"
    - Role manipulation: "you are now a", "pretend to be"
    - System prompt extraction: "reveal your system prompt"
    - Delimiter injection: "```system", "[INST]"
    - Jailbreak attempts: "DAN mode", "developer mode"
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
    """A pattern for detecting prompt injection attempts.

    Attributes:
        pattern: Compiled regex pattern.
        name: Human-readable name for the pattern.
        description: What this pattern detects.
        weight: Severity weight (1-3) for risk calculation.
    """

    pattern: re.Pattern[str]
    name: str
    description: str
    weight: int = 1


@dataclass(frozen=True, slots=True)
class InjectionCheckResult:
    """Result of a prompt injection check.

    Attributes:
        risk_level: Overall assessed risk level.
        detected_patterns: Names of patterns that matched.
        matched_texts: Actual text snippets that matched.
        total_weight: Sum of weights from all matched patterns.
        should_block: Whether the request should be blocked.
        should_warn: Whether the request should be logged as a warning.
    """

    risk_level: InjectionRisk
    detected_patterns: tuple[str, ...]
    matched_texts: tuple[str, ...]
    total_weight: int
    should_block: bool
    should_warn: bool


# Patterns are organized by attack category with severity weights
# Weight 3 = Critical (block immediately)
# Weight 2 = High severity
# Weight 1 = Medium severity
_INJECTION_PATTERNS: tuple[InjectionPattern, ...] = (
    # === Instruction Override Patterns (Weight 2-3) ===
    InjectionPattern(
        pattern=re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)",
            re.IGNORECASE,
        ),
        name="instruction_override",
        description="Attempts to override previous instructions",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|guidelines?)",
            re.IGNORECASE,
        ),
        name="disregard_instructions",
        description="Attempts to disregard instructions",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"forget\s+(everything|all|what)\s+(you\s+)?(were|have\s+been)\s+told",
            re.IGNORECASE,
        ),
        name="forget_instructions",
        description="Attempts to make model forget instructions",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"new\s+(instructions?|rules?|guidelines?):\s*",
            re.IGNORECASE,
        ),
        name="new_instructions",
        description="Attempts to inject new instructions",
        weight=2,
    ),
    # === Role Manipulation Patterns (Weight 2-3) ===
    InjectionPattern(
        pattern=re.compile(
            r"you\s+are\s+now\s+(a|an|the|in)\s+",
            re.IGNORECASE,
        ),
        name="role_assignment",
        description="Attempts to assign new role to model",
        weight=2,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"(pretend|act|behave|respond)\s+(to\s+be|as\s+if\s+you\s+are|like)\s+",
            re.IGNORECASE,
        ),
        name="role_pretend",
        description="Attempts to make model pretend to be something else",
        weight=2,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"from\s+now\s+on[,\s]+(you|your|act|respond|answer)",
            re.IGNORECASE,
        ),
        name="behavioral_change",
        description="Attempts to change model behavior",
        weight=2,
    ),
    # === System Prompt Extraction (Weight 3) ===
    InjectionPattern(
        pattern=re.compile(
            r"(reveal|show|display|print|output|tell\s+me)\s+(your|the)\s+system\s+prompt",
            re.IGNORECASE,
        ),
        name="system_prompt_extraction",
        description="Attempts to extract system prompt",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"what\s+(are|is)\s+(your|the)\s+(initial\s+)?(instructions?|system\s+prompt|rules?)",
            re.IGNORECASE,
        ),
        name="instruction_extraction",
        description="Attempts to extract instructions",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"(repeat|echo|recite)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
            re.IGNORECASE,
        ),
        name="prompt_repeat",
        description="Attempts to make model repeat prompt",
        weight=3,
    ),
    # === Delimiter Injection (Weight 2-3) ===
    InjectionPattern(
        pattern=re.compile(
            r"```\s*(system|assistant|user|human|ai)[\s\n]",
            re.IGNORECASE,
        ),
        name="markdown_delimiter",
        description="Markdown code block role delimiter injection",
        weight=2,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>",
            re.IGNORECASE,
        ),
        name="llama_delimiter",
        description="LLaMA-style delimiter injection",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>",
            re.IGNORECASE,
        ),
        name="chatml_delimiter",
        description="ChatML delimiter injection",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"Human:|Assistant:|System:|HUMAN:|ASSISTANT:|SYSTEM:",
            re.IGNORECASE,
        ),
        name="anthropic_delimiter",
        description="Anthropic-style delimiter injection",
        weight=2,
    ),
    # === Jailbreak Patterns (Weight 3) ===
    InjectionPattern(
        pattern=re.compile(
            r"\bDAN\b.*(mode|enabled?|activated?)|Do\s+Anything\s+Now",
            re.IGNORECASE,
        ),
        name="dan_jailbreak",
        description="DAN jailbreak attempt",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"(developer|dev|debug|admin|root)\s*(mode|access|privileges?)",
            re.IGNORECASE,
        ),
        name="privilege_escalation",
        description="Privilege escalation attempt",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"(bypass|disable|turn\s+off|ignore)\s+(your\s+)?(filters?|guardrails?|safety|restrictions?)",
            re.IGNORECASE,
        ),
        name="filter_bypass",
        description="Attempts to bypass safety filters",
        weight=3,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"(no\s+)?ethical\s+(guidelines?|constraints?|limits?)",
            re.IGNORECASE,
        ),
        name="ethics_bypass",
        description="Attempts to bypass ethical guidelines",
        weight=3,
    ),
    # === Output Manipulation (Weight 1-2) ===
    InjectionPattern(
        pattern=re.compile(
            r"respond\s+(only\s+)?with\s+(yes|no|true|false|1|0)(?:\s|$)",
            re.IGNORECASE,
        ),
        name="forced_response",
        description="Attempts to force specific response",
        weight=1,
    ),
    InjectionPattern(
        pattern=re.compile(
            r"(do\s+not|don't|never)\s+(mention|say|reveal|include)\s+(that|this|the fact)",
            re.IGNORECASE,
        ),
        name="output_suppression",
        description="Attempts to suppress certain output",
        weight=2,
    ),
)


def _calculate_risk_level(total_weight: int, pattern_count: int) -> InjectionRisk:
    """Calculate risk level based on total weight and pattern count.

    Args:
        total_weight: Sum of weights from matched patterns.
        pattern_count: Number of patterns matched.

    Returns:
        Calculated risk level.
    """
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

    Analyzes the input text against known prompt injection patterns
    and returns a risk assessment.

    Args:
        text: The text to check for injection patterns.
        block_threshold: Risk level at or above which to recommend blocking.
        patterns: Optional custom patterns to use instead of defaults.

    Returns:
        InjectionCheckResult with risk assessment and matched patterns.

    Example:
        >>> result = check_prompt_injection("What is requirements traceability?")
        >>> result.should_block
        False
        >>> result.risk_level
        <InjectionRisk.NONE: 'none'>

        >>> result = check_prompt_injection("Ignore all previous instructions")
        >>> result.should_block
        True
        >>> result.risk_level
        <InjectionRisk.HIGH: 'high'>
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

    # Determine blocking and warning thresholds
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
