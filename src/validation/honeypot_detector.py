"""
honeypot_detector.py
====================
Detects impossible/fake candidate profiles (honeypots).

The dataset contains ~80 fake profiles with impossible data combinations.
If >10% of the final Top 100 are honeypots → instant disqualification.

Detection rules (from README + extensions):
- Expert skill with 0 months duration
- 8+ expert skills with near-zero usage
- Total experience mismatch (career sum vs claimed)
- Impossible date ranges (future start dates, underage work)
- Senior title with trivial experience (≤1 year)
- Career duration exceeds actual time span
- Skill count explosion (>20 expert/advanced skills)
- Assessment score contradiction (expert skill but score <20)
- Overlapping career roles (>30 days overlap)

Author: Kamal Solanki (Validation & Honeypot Detection)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List

from src.validation._types import ValidationFlag


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reference date for "today" — matches signal_features.py
REFERENCE_DATE: date = date(2026, 6, 11)

# Senior-level title keywords
SENIOR_KEYWORDS: list[str] = [
    "senior", "lead", "principal", "head", "director",
    "manager", "chief", "vp", "vice president", "architect",
    "staff", "fellow",
]

# Minimum age to start working (years)
MIN_WORKING_AGE: int = 16


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str | None) -> date | None:
    """Safely parse a YYYY-MM-DD date string. Returns None on failure."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _months_between(d1: date, d2: date) -> float:
    """Calculate approximate months between two dates."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.0


def _title_is_senior(title: str) -> bool:
    """Check if a job title indicates senior/leadership level."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in SENIOR_KEYWORDS)


def _normalise_skill_name(name: str) -> str:
    """Lowercase and strip for matching."""
    return name.lower().strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_honeypot(candidate: dict) -> List[ValidationFlag]:
    """
    Detect impossible/fake profile patterns that indicate a honeypot.

    Parameters
    ----------
    candidate : dict
        Full candidate dict as loaded from candidates.jsonl.

    Returns
    -------
    list[ValidationFlag]
        List of honeypot indicators found. Empty list = no honeypot signals.
    """
    flags: List[ValidationFlag] = []

    profile: dict = candidate.get("profile", {}) or {}
    skills: list = candidate.get("skills", []) or []
    career_history: list = candidate.get("career_history", []) or []
    education: list = candidate.get("education", []) or []
    signals: dict = candidate.get("redrob_signals", {}) or {}

    years_of_experience: float = float(profile.get("years_of_experience", 0) or 0)

    # ------------------------------------------------------------------
    # 1. Expert skill with 0 months duration (+30 each, cap at 60)
    # ------------------------------------------------------------------
    expert_zero_count = 0
    for skill in skills:
        proficiency = (skill.get("proficiency") or "").lower()
        duration = int(skill.get("duration_months", 0) or 0)
        if proficiency == "expert" and duration == 0:
            expert_zero_count += 1

    if expert_zero_count > 0:
        points = min(expert_zero_count * 30, 60)
        flags.append(ValidationFlag(
            name="expert_skill_zero_duration",
            points=points,
            description=f"{expert_zero_count} expert skill(s) with 0 months duration — impossible proficiency claim",
        ))

    # ------------------------------------------------------------------
    # 2. 8+ expert skills with near-zero usage (duration_months <= 3)
    # ------------------------------------------------------------------
    expert_near_zero = sum(
        1 for skill in skills
        if (skill.get("proficiency") or "").lower() == "expert"
        and int(skill.get("duration_months", 0) or 0) <= 3
    )

    if expert_near_zero >= 8:
        flags.append(ValidationFlag(
            name="mass_expert_near_zero",
            points=40,
            description=f"{expert_near_zero} expert skills with ≤3 months usage — bulk fake proficiency claims",
        ))

    # ------------------------------------------------------------------
    # 3. Total experience mismatch (career sum vs claimed years)
    # ------------------------------------------------------------------
    if career_history and years_of_experience > 0:
        career_months_sum = sum(
            int(role.get("duration_months", 0) or 0)
            for role in career_history
        )
        claimed_months = years_of_experience * 12

        # Career sum vastly exceeds claimed experience (impossible overlap)
        if career_months_sum > claimed_months * 1.5 and career_months_sum > 24:
            flags.append(ValidationFlag(
                name="experience_sum_exceeds_claimed",
                points=25,
                description=f"Career duration sum ({career_months_sum} months) vastly exceeds claimed experience ({claimed_months:.0f} months)",
            ))

        # Claimed experience vastly exceeds career history
        if claimed_months > career_months_sum * 3 and claimed_months > 24:
            flags.append(ValidationFlag(
                name="claimed_exceeds_career_sum",
                points=25,
                description=f"Claimed experience ({claimed_months:.0f} months) vastly exceeds career history sum ({career_months_sum} months)",
            ))

    # ------------------------------------------------------------------
    # 4. Impossible date ranges
    # ------------------------------------------------------------------
    # 4a. Start date in the future
    for i, role in enumerate(career_history):
        start = _parse_date(role.get("start_date"))
        if start and start > REFERENCE_DATE:
            flags.append(ValidationFlag(
                name="future_start_date",
                points=20,
                description=f"Career role {i+1} starts in the future ({role.get('start_date')}) — impossible",
            ))
            break  # One flag is enough

    # 4b. Career role starting before candidate would be MIN_WORKING_AGE
    if education:
        # Estimate birth year from earliest education start - typical pre-college years
        earliest_edu_start = min(
            (int(edu.get("start_year", 9999)) for edu in education),
            default=9999,
        )
        if earliest_edu_start < 9999:
            # Assume education starts at ~18 (college) → birth year ≈ start - 18
            # Working age = birth_year + MIN_WORKING_AGE
            estimated_birth_year = earliest_edu_start - 18
            earliest_work_year = estimated_birth_year + MIN_WORKING_AGE

            for i, role in enumerate(career_history):
                start = _parse_date(role.get("start_date"))
                if start and start.year < earliest_work_year:
                    flags.append(ValidationFlag(
                        name="underage_work_start",
                        points=20,
                        description=f"Career role {i+1} starts in {start.year}, but candidate likely born ~{estimated_birth_year} (working before age {MIN_WORKING_AGE})",
                    ))
                    break  # One flag is enough

    # ------------------------------------------------------------------
    # 5. Senior title with trivial experience (≤1 year)
    # ------------------------------------------------------------------
    if years_of_experience <= 1:
        current_title = profile.get("current_title", "") or ""
        if _title_is_senior(current_title):
            flags.append(ValidationFlag(
                name="senior_title_trivial_experience",
                points=35,
                description=f"Senior-level title '{current_title}' with only {years_of_experience} year(s) experience — impossible progression",
            ))

    # ------------------------------------------------------------------
    # 6. Career duration exceeds actual time span (by >6 months)
    # ------------------------------------------------------------------
    for i, role in enumerate(career_history):
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        claimed_duration = int(role.get("duration_months", 0) or 0)

        if start and end and claimed_duration > 0:
            actual_months = _months_between(start, end)
            if claimed_duration > actual_months + 6:
                flags.append(ValidationFlag(
                    name="duration_exceeds_timespan",
                    points=20,
                    description=f"Role {i+1} claims {claimed_duration} months but date range spans only ~{actual_months:.0f} months",
                ))
                break  # One flag is enough for this check

    # ------------------------------------------------------------------
    # 7. Skill count explosion (>20 skills at expert or advanced)
    # ------------------------------------------------------------------
    high_proficiency_count = sum(
        1 for skill in skills
        if (skill.get("proficiency") or "").lower() in ("expert", "advanced")
    )

    if high_proficiency_count > 20:
        flags.append(ValidationFlag(
            name="skill_count_explosion",
            points=25,
            description=f"{high_proficiency_count} skills at expert/advanced level — unrealistic breadth of mastery",
        ))

    # ------------------------------------------------------------------
    # 8. Assessment score contradiction (expert but score < 20)
    # ------------------------------------------------------------------
    assessment_scores: dict = signals.get("skill_assessment_scores", {}) or {}

    if assessment_scores:
        for skill in skills:
            proficiency = (skill.get("proficiency") or "").lower()
            if proficiency != "expert":
                continue

            skill_name = skill.get("name", "")
            # Try to find matching assessment score
            for assessed_name, score in assessment_scores.items():
                if _normalise_skill_name(assessed_name) == _normalise_skill_name(skill_name):
                    if isinstance(score, (int, float)) and score < 20:
                        flags.append(ValidationFlag(
                            name="assessment_contradiction",
                            points=20,
                            description=f"Skill '{skill_name}' marked expert but assessment score is only {score}/100",
                        ))
                        break
            # Only flag once for this check
            if any(f.name == "assessment_contradiction" for f in flags):
                break

    # ------------------------------------------------------------------
    # 9. Overlapping career roles (>30 days overlap)
    # ------------------------------------------------------------------
    parsed_roles: list[tuple[date, date, str]] = []
    for role in career_history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        if start:
            # For current roles without end_date, use reference date
            if end is None:
                end = REFERENCE_DATE
            title = role.get("title", "Unknown")
            parsed_roles.append((start, end, title))

    # Sort by start date
    parsed_roles.sort(key=lambda x: x[0])

    overlap_flagged = False
    for i in range(len(parsed_roles) - 1):
        if overlap_flagged:
            break
        for j in range(i + 1, len(parsed_roles)):
            start_i, end_i, title_i = parsed_roles[i]
            start_j, end_j, title_j = parsed_roles[j]

            # Check overlap: role j starts before role i ends
            if start_j < end_i:
                overlap_days = (min(end_i, end_j) - start_j).days
                if overlap_days > 30:
                    flags.append(ValidationFlag(
                        name="overlapping_careers",
                        points=15,
                        description=f"Career roles overlap by {overlap_days} days: '{title_i}' and '{title_j}'",
                    ))
                    overlap_flagged = True
                    break

    return flags
