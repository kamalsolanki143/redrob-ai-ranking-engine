"""
skill_features.py
=================
Computes the `skill_score` (0.0 – 1.0) for a single candidate
relative to a job description's required skill list.

Score breakdown (weights sum to 1.0):
  - direct_match_score  : 40%   how many JD skills the candidate has
  - proficiency_weight  : 25%   how proficient they are in matched skills
  - assessment_bonus    : 20%   platform skill-test scores for matched skills
  - duration_bonus      : 15%   how long they've used the matched skills

Author : Muskan (Feature Engineering)
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFICIENCY_MAP: dict[str, float] = {
    "beginner":     0.25,
    "intermediate": 0.50,
    "advanced":     0.75,
    "expert":       1.00,
}

# Duration considered "full marks" for a skill (3 years = 36 months)
DURATION_FULL_MONTHS: float = 36.0

# Component weights inside skill_score — must sum to 1.0
WEIGHT_DIRECT_MATCH: float  = 0.40
WEIGHT_PROFICIENCY:  float  = 0.25
WEIGHT_ASSESSMENT:   float  = 0.20
WEIGHT_DURATION:     float  = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase and strip punctuation for fuzzy-ish matching."""
    return re.sub(r"[^a-z0-9 ]", " ", name.lower()).strip()


def _skills_match(candidate_skill_name: str, jd_skill: str) -> bool:
    """
    Return True if a candidate's skill name 'matches' a JD skill.

    Strategy:
      1. Exact normalised match.
      2. Either name is a substring of the other (handles 'ML' vs
         'Machine Learning', 'TF' vs 'TensorFlow', abbreviations, etc.)
    """
    c = _normalise(candidate_skill_name)
    j = _normalise(jd_skill)
    if c == j:
        return True
    # substring check (both directions)
    if c in j or j in c:
        return True
    return False


def _find_matched_skills(
    candidate_skills: list[dict],
    jd_skills: list[str],
) -> list[dict]:
    """
    Return the subset of candidate skill objects that match at least one
    JD skill.  De-duplicates: a candidate skill is counted at most once.
    """
    matched: list[dict] = []
    for skill in candidate_skills:
        skill_name: str = skill.get("name", "")
        if not skill_name:
            continue
        for jd_skill in jd_skills:
            if _skills_match(skill_name, jd_skill):
                matched.append(skill)
                break  # avoid double-counting the same candidate skill
    return matched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_skill_score(
    candidate: dict,
    jd_skills: list[str],
) -> float:
    """
    Compute the normalised skill score (0.0 – 1.0) for a candidate.

    Parameters
    ----------
    candidate  : Full candidate dict (as loaded from candidates.jsonl).
    jd_skills  : List of skill-name strings from the job description,
                 e.g. ["Python", "Machine Learning", "SQL", "TensorFlow"].

    Returns
    -------
    float in [0.0, 1.0]
    """
    # Defensive: handle missing or malformed data gracefully
    if not jd_skills:
        return 0.0

    candidate_skills: list[dict] = candidate.get("skills", []) or []
    signals: dict = candidate.get("redrob_signals", {}) or {}

    # ------------------------------------------------------------------
    # 1. direct_match_score  (40 %)
    # ------------------------------------------------------------------
    matched_skills = _find_matched_skills(candidate_skills, jd_skills)
    n_matched = len(matched_skills)
    n_jd      = len(jd_skills)

    direct_match_score: float = min(n_matched / n_jd, 1.0) if n_jd > 0 else 0.0

    # ------------------------------------------------------------------
    # 2. proficiency_weight  (25 %)
    #    Average proficiency level of matched skills.
    # ------------------------------------------------------------------
    if n_matched > 0:
        prof_values = [
            PROFICIENCY_MAP.get(s.get("proficiency", "").lower(), 0.0)
            for s in matched_skills
        ]
        proficiency_weight: float = sum(prof_values) / len(prof_values)
    else:
        proficiency_weight = 0.0

    # ------------------------------------------------------------------
    # 3. assessment_bonus  (20 %)
    #    Use Redrob platform assessment scores for matched skills.
    #    skill_assessment_scores is a dict {skill_name: 0–100}.
    #    Can be empty {}; we normalise to [0, 1].
    # ------------------------------------------------------------------
    assessment_scores_raw: dict = signals.get("skill_assessment_scores", {}) or {}

    # Build a normalised-key lookup for assessment scores
    assessment_lookup: dict[str, float] = {
        _normalise(k): v / 100.0
        for k, v in assessment_scores_raw.items()
        if isinstance(v, (int, float))
    }

    assessment_values: list[float] = []
    for skill in matched_skills:
        key = _normalise(skill.get("name", ""))
        if key in assessment_lookup:
            assessment_values.append(assessment_lookup[key])

    assessment_bonus: float = (
        sum(assessment_values) / len(assessment_values)
        if assessment_values
        else 0.0
    )

    # ------------------------------------------------------------------
    # 4. duration_bonus  (15 %)
    #    Average months of experience in matched skills, capped at
    #    DURATION_FULL_MONTHS (36 months = 3 years → score 1.0).
    #    duration_months may be absent; default to 0.
    # ------------------------------------------------------------------
    duration_values: list[float] = [
        float(s.get("duration_months", 0) or 0)
        for s in matched_skills
    ]
    if duration_values:
        avg_duration = sum(duration_values) / len(duration_values)
        duration_bonus: float = min(avg_duration / DURATION_FULL_MONTHS, 1.0)
    else:
        duration_bonus = 0.0

    # ------------------------------------------------------------------
    # Weighted combination → skill_score
    # ------------------------------------------------------------------
    skill_score: float = (
        WEIGHT_DIRECT_MATCH * direct_match_score
        + WEIGHT_PROFICIENCY  * proficiency_weight
        + WEIGHT_ASSESSMENT   * assessment_bonus
        + WEIGHT_DURATION     * duration_bonus
    )

    # Clamp to [0.0, 1.0] to guard against any floating-point oddity
    return float(max(0.0, min(1.0, skill_score)))
