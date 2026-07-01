"""
profile_validator.py
====================
Structural integrity checks for candidate profiles.

Detects broken, invalid, or malformed profile data including:
- Missing critical fields
- Invalid date formats
- Negative durations
- Broken career history timelines
- Duplicate entries
- Inverted salary ranges

Each violation produces a ValidationFlag with a point value that
contributes to the candidate's overall risk_score.

Author: Kamal Solanki (Validation & Honeypot Detection)
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from src.validation._types import ValidationFlag


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Critical fields that must exist for a valid profile
REQUIRED_PROFILE_FIELDS: list[str] = ["headline", "years_of_experience", "current_title"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str | None) -> datetime | None:
    """Safely parse a YYYY-MM-DD date string. Returns None on failure."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_profile(candidate: dict) -> List[ValidationFlag]:
    """
    Run structural integrity checks on a candidate profile.

    Parameters
    ----------
    candidate : dict
        Full candidate dict as loaded from candidates.jsonl.

    Returns
    -------
    list[ValidationFlag]
        List of validation violations found. Empty list = structurally valid.
    """
    flags: List[ValidationFlag] = []

    # ------------------------------------------------------------------
    # 1. Missing candidate_id
    # ------------------------------------------------------------------
    if not candidate.get("candidate_id"):
        flags.append(ValidationFlag(
            name="missing_candidate_id",
            points=10,
            description="Missing candidate_id field",
        ))

    # ------------------------------------------------------------------
    # 2. Missing critical profile fields
    # ------------------------------------------------------------------
    profile: dict = candidate.get("profile", {}) or {}

    for field_name in REQUIRED_PROFILE_FIELDS:
        value = profile.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            flags.append(ValidationFlag(
                name=f"missing_profile_{field_name}",
                points=10,
                description=f"Missing critical profile field: {field_name}",
            ))

    # ------------------------------------------------------------------
    # 3. Empty career history
    # ------------------------------------------------------------------
    career_history: list = candidate.get("career_history", []) or []

    if not career_history:
        flags.append(ValidationFlag(
            name="empty_career_history",
            points=10,
            description="Career history is empty — no work experience records",
        ))

    # ------------------------------------------------------------------
    # 4. Years of experience out of range
    # ------------------------------------------------------------------
    years_of_experience = profile.get("years_of_experience")
    if years_of_experience is not None:
        try:
            yoe = float(years_of_experience)
            if yoe < 0:
                flags.append(ValidationFlag(
                    name="negative_experience",
                    points=20,
                    description=f"Negative years of experience: {yoe}",
                ))
            elif yoe > 50:
                flags.append(ValidationFlag(
                    name="excessive_experience",
                    points=20,
                    description=f"Unrealistic years of experience: {yoe} (>50 years)",
                ))
        except (TypeError, ValueError):
            flags.append(ValidationFlag(
                name="invalid_experience_value",
                points=10,
                description="years_of_experience is not a valid number",
            ))

    # ------------------------------------------------------------------
    # 5. Career history date and duration checks
    # ------------------------------------------------------------------
    for i, role in enumerate(career_history):
        # Negative duration_months
        duration_months = role.get("duration_months")
        if duration_months is not None:
            try:
                dm = int(duration_months)
                if dm < 0:
                    flags.append(ValidationFlag(
                        name="negative_duration",
                        points=15,
                        description=f"Negative duration_months ({dm}) in career role {i+1}: {role.get('title', 'Unknown')} at {role.get('company', 'Unknown')}",
                    ))
            except (TypeError, ValueError):
                pass

        # Invalid date formats
        start_str = role.get("start_date")
        end_str = role.get("end_date")

        start_date = _parse_date(start_str)
        end_date = _parse_date(end_str)

        if start_str and start_date is None:
            flags.append(ValidationFlag(
                name="invalid_start_date",
                points=10,
                description=f"Invalid start_date format in career role {i+1}: '{start_str}'",
            ))

        if end_str and end_date is None:
            flags.append(ValidationFlag(
                name="invalid_end_date",
                points=10,
                description=f"Invalid end_date format in career role {i+1}: '{end_str}'",
            ))

        # end_date before start_date
        if start_date and end_date and end_date < start_date:
            flags.append(ValidationFlag(
                name="end_before_start",
                points=15,
                description=f"end_date ({end_str}) before start_date ({start_str}) in career role {i+1}: {role.get('title', 'Unknown')} at {role.get('company', 'Unknown')}",
            ))

    # ------------------------------------------------------------------
    # 6. Education year checks
    # ------------------------------------------------------------------
    education: list = candidate.get("education", []) or []

    for i, edu in enumerate(education):
        start_year = edu.get("start_year")
        end_year = edu.get("end_year")

        if start_year is not None and end_year is not None:
            try:
                sy = int(start_year)
                ey = int(end_year)
                if ey < sy:
                    flags.append(ValidationFlag(
                        name="education_year_inverted",
                        points=10,
                        description=f"Education end_year ({ey}) before start_year ({sy}) at {edu.get('institution', 'Unknown')}",
                    ))
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------------
    # 7. Duplicate career entries
    # ------------------------------------------------------------------
    seen_roles: set = set()
    for role in career_history:
        key = (
            (role.get("company") or "").lower().strip(),
            (role.get("title") or "").lower().strip(),
            (role.get("start_date") or "").strip(),
        )
        if key[0] and key[1] and key[2]:  # Only check if all fields present
            if key in seen_roles:
                flags.append(ValidationFlag(
                    name="duplicate_career_entry",
                    points=15,
                    description=f"Duplicate career entry: {role.get('title')} at {role.get('company')} starting {role.get('start_date')}",
                ))
            else:
                seen_roles.add(key)

    # ------------------------------------------------------------------
    # 8. Salary range inverted
    # ------------------------------------------------------------------
    signals: dict = candidate.get("redrob_signals", {}) or {}
    salary_range = signals.get("expected_salary_range_inr_lpa", {}) or {}

    if salary_range:
        sal_min = salary_range.get("min")
        sal_max = salary_range.get("max")
        if sal_min is not None and sal_max is not None:
            try:
                if float(sal_min) > float(sal_max):
                    flags.append(ValidationFlag(
                        name="salary_range_inverted",
                        points=5,
                        description=f"Salary range inverted: min ({sal_min}) > max ({sal_max}) LPA",
                    ))
            except (TypeError, ValueError):
                pass

    return flags
