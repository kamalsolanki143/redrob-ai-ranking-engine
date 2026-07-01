"""
validation/_types.py
====================
Shared data types for the validation module.

Separated from __init__.py to avoid circular imports when
sub-modules (profile_validator, honeypot_detector, anomaly_checker)
need to reference ValidationFlag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Constants — Tier Thresholds
# ---------------------------------------------------------------------------

HONEYPOT_THRESHOLD: int = 80
SUSPICIOUS_THRESHOLD: int = 50


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationFlag:
    """A single validation rule violation."""
    name: str
    points: int
    description: str


@dataclass
class ValidationResult:
    """Complete validation verdict for a single candidate."""
    candidate_id: str
    is_honeypot: bool
    risk_score: int
    primary_reason: str
    flags: List[ValidationFlag] = field(default_factory=list)
    tier: str = "clean"
