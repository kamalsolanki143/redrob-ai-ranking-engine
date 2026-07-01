"""
anomaly_checker.py
==================
Detects behavioral and statistical anomalies in candidate profiles.

Identifies suspicious patterns that suggest fabricated or inflated profiles:
- Suspicious career jumps (intern → Director in <2 years)
- Unrealistic growth (<3 years but senior title)
- Fake activity patterns (mass applications, zero responses)
- Endorsement anomalies (high endorsements on 0-month skills)
- Profile completeness vs content mismatch
- Impossible signup/activity dates
- Certification year anomalies
- Title-description mismatch

Author: Kamal Solanki (Validation & Honeypot Detection)
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from src.validation._types import ValidationFlag


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUNIOR_KEYWORDS: list[str] = [
    "intern", "trainee", "junior", "associate", "assistant", "entry",
]

SENIOR_KEYWORDS: list[str] = [
    "senior", "lead", "principal", "director", "vp",
    "vice president", "chief", "architect", "head", "staff", "fellow",
]

TECHNICAL_TITLE_KEYWORDS: list[str] = [
    "engineer", "developer", "architect", "scientist",
    "programmer", "coder", "devops", "sre", "mlops",
]

TECHNICAL_DESCRIPTION_KEYWORDS: list[str] = [
    "code", "system", "data", "build", "deploy", "api", "model",
    "software", "algorithm", "database", "server", "cloud",
    "pipeline", "architecture", "framework", "testing", "debug",
    "infrastructure", "machine learning", "neural", "training",
]


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


def _title_has_keyword(title: str, keywords: list[str]) -> bool:
    """Check if title contains any of the keywords (case-insensitive)."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in keywords)


def _months_between_dates(d1: datetime, d2: datetime) -> float:
    """Calculate approximate months between two datetimes."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_anomalies(candidate: dict) -> List[ValidationFlag]:
    """
    Detect behavioral and statistical anomalies in a candidate profile.

    Parameters
    ----------
    candidate : dict
        Full candidate dict as loaded from candidates.jsonl.

    Returns
    -------
    list[ValidationFlag]
        List of anomaly flags found. Empty list = no anomalies detected.
    """
    flags: List[ValidationFlag] = []

    profile: dict = candidate.get("profile", {}) or {}
    skills: list = candidate.get("skills", []) or []
    career_history: list = candidate.get("career_history", []) or []
    education: list = candidate.get("education", []) or []
    certifications: list = candidate.get("certifications", []) or []
    signals: dict = candidate.get("redrob_signals", {}) or {}

    years_of_experience: float = float(profile.get("years_of_experience", 0) or 0)
    current_title: str = profile.get("current_title", "") or ""

    # ------------------------------------------------------------------
    # 1. Suspicious career jumps
    #    Junior → Senior in one consecutive jump with < 2 years gap
    # ------------------------------------------------------------------
    if len(career_history) >= 2:
        # Sort career history by start_date
        sorted_roles = sorted(
            career_history,
            key=lambda r: r.get("start_date", "0000-00-00"),
        )

        for i in range(len(sorted_roles) - 1):
            prev_title = sorted_roles[i].get("title", "") or ""
            next_title = sorted_roles[i + 1].get("title", "") or ""

            if _title_has_keyword(prev_title, JUNIOR_KEYWORDS) and \
               _title_has_keyword(next_title, SENIOR_KEYWORDS):
                # Check time gap between roles
                prev_end = _parse_date(sorted_roles[i].get("end_date"))
                next_start = _parse_date(sorted_roles[i + 1].get("start_date"))

                if prev_end and next_start:
                    gap_months = _months_between_dates(prev_end, next_start)
                    # Include the duration of the junior role too
                    prev_duration = int(sorted_roles[i].get("duration_months", 0) or 0)
                    total_time = prev_duration + gap_months

                    if total_time < 24:  # Less than 2 years total
                        flags.append(ValidationFlag(
                            name="suspicious_career_jump",
                            points=20,
                            description=f"Suspicious jump from '{prev_title}' to '{next_title}' in <2 years — unrealistic progression",
                        ))
                        break  # One flag is enough

    # ------------------------------------------------------------------
    # 2. Unrealistic growth
    #    <3 years experience but current title is senior-level
    # ------------------------------------------------------------------
    if years_of_experience < 3 and _title_has_keyword(current_title, SENIOR_KEYWORDS):
        flags.append(ValidationFlag(
            name="unrealistic_growth",
            points=15,
            description=f"Only {years_of_experience} years experience but holds senior title '{current_title}'",
        ))

    # ------------------------------------------------------------------
    # 3. Fake activity pattern
    #    High applications but zero response and zero interview completion
    # ------------------------------------------------------------------
    apps_submitted = int(signals.get("applications_submitted_30d", 0) or 0)
    response_rate = float(signals.get("recruiter_response_rate", 0) or 0)
    interview_rate = float(signals.get("interview_completion_rate", 0) or 0)

    if apps_submitted > 20 and response_rate == 0 and interview_rate == 0:
        flags.append(ValidationFlag(
            name="fake_activity_pattern",
            points=15,
            description=f"Submitted {apps_submitted} applications but 0% response rate and 0% interview completion — bot-like behavior",
        ))

    # ------------------------------------------------------------------
    # 4. Endorsement anomaly
    #    Skill with 0 months duration but >50 endorsements
    # ------------------------------------------------------------------
    for skill in skills:
        duration = int(skill.get("duration_months", 0) or 0)
        endorsements = int(skill.get("endorsements", 0) or 0)

        if duration == 0 and endorsements > 50:
            flags.append(ValidationFlag(
                name="endorsement_anomaly",
                points=10,
                description=f"Skill '{skill.get('name', 'Unknown')}' has {endorsements} endorsements but 0 months experience — fake endorsements",
            ))
            break  # One flag is enough

    # ------------------------------------------------------------------
    # 5. Profile completeness vs content mismatch
    #    Completeness >90% but no skills and no career history
    # ------------------------------------------------------------------
    completeness = float(signals.get("profile_completeness_score", 0) or 0)

    if completeness > 90 and len(skills) == 0 and len(career_history) == 0:
        flags.append(ValidationFlag(
            name="completeness_content_mismatch",
            points=20,
            description=f"Profile completeness is {completeness}% but has 0 skills and empty career history — data inconsistency",
        ))

    # ------------------------------------------------------------------
    # 6. Signup-to-active impossible
    #    last_active_date < signup_date
    # ------------------------------------------------------------------
    signup_date = _parse_date(signals.get("signup_date"))
    last_active_date = _parse_date(signals.get("last_active_date"))

    if signup_date and last_active_date and last_active_date < signup_date:
        flags.append(ValidationFlag(
            name="active_before_signup",
            points=15,
            description=f"Last active ({signals.get('last_active_date')}) before signup ({signals.get('signup_date')}) — impossible timeline",
        ))

    # ------------------------------------------------------------------
    # 7. Certification year anomaly
    #    Certification year in the future (>2026) or before education start
    # ------------------------------------------------------------------
    if certifications:
        earliest_edu_start = None
        if education:
            edu_years = [int(edu.get("start_year", 9999)) for edu in education
                        if edu.get("start_year") is not None]
            if edu_years:
                earliest_edu_start = min(edu_years)

        for cert in certifications:
            cert_year = cert.get("year")
            if cert_year is not None:
                try:
                    cy = int(cert_year)
                    if cy > 2026:
                        flags.append(ValidationFlag(
                            name="future_certification",
                            points=10,
                            description=f"Certification '{cert.get('name', 'Unknown')}' year {cy} is in the future",
                        ))
                        break
                    if earliest_edu_start and cy < earliest_edu_start:
                        flags.append(ValidationFlag(
                            name="premature_certification",
                            points=10,
                            description=f"Certification '{cert.get('name', 'Unknown')}' year {cy} is before education start ({earliest_edu_start})",
                        ))
                        break
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------
    # 8. Title-description mismatch heuristic
    #    Technical title but description has zero technical keywords
    # ------------------------------------------------------------------
    if career_history:
        # Check current/most recent role
        current_roles = [r for r in career_history if r.get("is_current", False)]
        role_to_check = current_roles[0] if current_roles else career_history[0]

        role_title = role_to_check.get("title", "") or ""
        role_description = (role_to_check.get("description", "") or "").lower()

        if _title_has_keyword(role_title, TECHNICAL_TITLE_KEYWORDS):
            # Check if description contains ANY technical keyword
            has_technical_content = any(
                kw in role_description
                for kw in TECHNICAL_DESCRIPTION_KEYWORDS
            )
            if not has_technical_content and len(role_description) > 50:
                flags.append(ValidationFlag(
                    name="title_description_mismatch",
                    points=10,
                    description=f"Technical title '{role_title}' but role description contains no technical keywords — possible fake title",
                ))

    return flags
