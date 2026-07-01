"""
validation/__init__.py
======================
Unified Candidate Validation & Honeypot Detection System.

Provides a single entry point to validate any candidate profile
and determine whether it's a honeypot (fake), suspicious, or clean.

Multi-tier risk scoring (additive points, 0–100):
  - risk_score >= 80  → "honeypot"   → final_score = 0.0 (excluded from Top 100)
  - risk_score >= 50  → "suspicious" → final_score *= 0.7 (penalized)
  - risk_score <  50  → "clean"      → no penalty

Usage:
    from src.validation import validate_candidate, validate_candidates_batch
    from src.validation import get_penalty_multiplier, ValidationResult

    result = validate_candidate(candidate_dict)
    print(result.tier, result.risk_score, result.primary_reason)

    multiplier = get_penalty_multiplier(result)
    final_score *= multiplier

Author: Kamal Solanki (Validation & Honeypot Detection)
"""

from __future__ import annotations

from typing import List

import pandas as pd

# Import shared types from _types module (avoids circular imports)
from src.validation._types import (
    ValidationFlag,
    ValidationResult,
    HONEYPOT_THRESHOLD,
    SUSPICIOUS_THRESHOLD,
)

# Import sub-module functions
from src.validation.profile_validator import validate_profile
from src.validation.honeypot_detector import detect_honeypot
from src.validation.anomaly_checker import check_anomalies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_tier(risk_score: int) -> str:
    """Determine risk tier from score."""
    if risk_score >= HONEYPOT_THRESHOLD:
        return "honeypot"
    if risk_score >= SUSPICIOUS_THRESHOLD:
        return "suspicious"
    return "clean"


def get_penalty_multiplier(result: ValidationResult) -> float:
    """
    Return the scoring multiplier based on validation tier.

    - honeypot   → 0.0 (completely excluded)
    - suspicious → 0.7 (penalized)
    - clean      → 1.0 (no penalty)
    """
    if result.tier == "honeypot":
        return 0.0
    if result.tier == "suspicious":
        return 0.7
    return 1.0


# ---------------------------------------------------------------------------
# Unified Validation API
# ---------------------------------------------------------------------------

def validate_candidate(candidate: dict) -> ValidationResult:
    """
    Run all validation checks on a single candidate and produce a verdict.

    Combines profile validation, honeypot detection, and anomaly checking
    into a single ValidationResult with additive risk scoring.

    Parameters
    ----------
    candidate : dict
        Full candidate dict as loaded from candidates.jsonl.

    Returns
    -------
    ValidationResult
        Complete validation verdict including tier, risk_score, and flags.
    """
    candidate_id: str = candidate.get("candidate_id", "UNKNOWN")

    # Run all three validation modules
    all_flags: List[ValidationFlag] = []
    all_flags.extend(validate_profile(candidate))
    all_flags.extend(detect_honeypot(candidate))
    all_flags.extend(check_anomalies(candidate))

    # Compute risk score (additive, capped at 100)
    raw_score = sum(f.points for f in all_flags)
    risk_score = min(raw_score, 100)

    # Determine tier and honeypot status
    tier = compute_tier(risk_score)
    is_honeypot = (tier == "honeypot")

    # Primary reason = highest-point flag description
    if all_flags:
        primary_flag = max(all_flags, key=lambda f: f.points)
        primary_reason = primary_flag.description
    else:
        primary_reason = "Clean profile"

    return ValidationResult(
        candidate_id=candidate_id,
        is_honeypot=is_honeypot,
        risk_score=risk_score,
        primary_reason=primary_reason,
        flags=all_flags,
        tier=tier,
    )


def validate_candidates_batch(candidates: list[dict]) -> pd.DataFrame:
    """
    Validate a batch of candidates and return results as a DataFrame.

    Parameters
    ----------
    candidates : list[dict]
        List of candidate dicts (typically the FAISS-filtered top ~1000).

    Returns
    -------
    pd.DataFrame
        Columns: candidate_id, is_honeypot, risk_score, tier, primary_reason, flags
        The 'flags' column contains comma-separated flag names.
    """
    records: list[dict] = []

    for candidate in candidates:
        result = validate_candidate(candidate)
        records.append({
            "candidate_id": result.candidate_id,
            "is_honeypot": result.is_honeypot,
            "risk_score": result.risk_score,
            "tier": result.tier,
            "primary_reason": result.primary_reason,
            "flags": ", ".join(f.name for f in result.flags) if result.flags else "",
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ValidationFlag",
    "ValidationResult",
    "HONEYPOT_THRESHOLD",
    "SUSPICIOUS_THRESHOLD",
    "compute_tier",
    "get_penalty_multiplier",
    "validate_candidate",
    "validate_candidates_batch",
    "validate_profile",
    "detect_honeypot",
    "check_anomalies",
]
