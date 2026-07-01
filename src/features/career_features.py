"""
career_features.py
==================
Computes the `career_score` (0.0 – 1.0) for a single candidate
relative to a job description's experience and industry requirements.

Score breakdown (weights sum to 1.0):
  - experience_score     : 30%   raw years vs JD minimum requirement
  - progression_score    : 25%   career level / seniority trajectory
  - tenure_score         : 20%   avg time per role (stability indicator)
  - industry_match_score : 15%   does career history align with JD industry?
  - education_score      : 10%   institution tier

Author : Muskan (Feature Engineering)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keywords that signal a senior-level role
SENIOR_KEYWORDS: list[str] = [
    "senior", "lead", "principal", "head", "director",
    "manager", "chief", "vp", "vice president", "architect",
    "staff", "fellow",
]

# Keywords that signal a junior-level role
JUNIOR_KEYWORDS: list[str] = [
    "junior", "intern", "internship", "trainee",
    "associate", "assistant", "entry",
]

# Institution prestige tier → score
TIER_MAP: dict[str, float] = {
    "tier_1": 1.00,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
    "unknown": 0.25,  # treat unknown same as lowest known tier
}

# Tenure thresholds (months)
TENURE_VERY_LOW:  float = 6.0
TENURE_LOW:       float = 12.0
TENURE_HIGH:      float = 36.0

# Component weights inside career_score — must sum to 1.0
WEIGHT_EXPERIENCE:   float = 0.30
WEIGHT_PROGRESSION:  float = 0.25
WEIGHT_TENURE:       float = 0.20
WEIGHT_INDUSTRY:     float = 0.15
WEIGHT_EDUCATION:    float = 0.10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contains_keyword(title: str, keywords: list[str]) -> bool:
    """Case-insensitive check whether a job title contains any keyword."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in keywords)


def _compute_experience_score(
    years_of_experience: float,
    jd_min_experience: float,
) -> float:
    """
    Linear scale: meets minimum → 1.0; zero XP → 0.0.
    Candidates above the minimum are not penalised.
    """
    yoe = max(0.0, float(years_of_experience or 0))
    req = max(0.0, float(jd_min_experience or 0))

    if req <= 0:
        # No minimum stated — anyone with any experience scores 1.0,
        # complete newcomers score 0.5 (neutral).
        return 1.0 if yoe > 0 else 0.5

    if yoe >= req:
        return 1.0
    if yoe <= 0:
        return 0.0
    return yoe / req


def _compute_progression_score(career_history: list[dict]) -> float:
    """
    Assess career seniority from the most recent job title.

    Heuristic:
      - Most recent title has a senior keyword  → 1.0
      - Most recent title has a junior keyword  → 0.2
      - Neither (mid-level)                     → 0.6
    """
    if not career_history:
        return 0.3  # No career history at all → pessimistic default

    # Sort by is_current first, then by start_date descending
    # so the current/most-recent role is first
    sorted_history: list[dict] = sorted(
        career_history,
        key=lambda x: (
            1 if x.get("is_current", False) else 0,
            str(x.get("start_date", "0000-00-00")),
        ),
        reverse=True,
    )

    most_recent_title: str = sorted_history[0].get("title", "")

    if _contains_keyword(most_recent_title, SENIOR_KEYWORDS):
        return 1.0
    if _contains_keyword(most_recent_title, JUNIOR_KEYWORDS):
        return 0.2
    return 0.6


def _compute_tenure_score(career_history: list[dict]) -> float:
    """
    Average tenure per role expressed as a score.

    < 6 months  → 0.1  (job-hopper red flag)
    6–12 months → 0.5  (somewhat short but ok)
    12–36 months→ 1.0  (healthy / ideal)
    > 36 months → 0.8  (stable, slightly rigid)
    """
    if not career_history:
        return 0.5  # Unknown → neutral

    durations: list[float] = [
        float(role.get("duration_months", 0) or 0)
        for role in career_history
    ]
    # Filter out zero-duration roles (data anomaly / current role still in progress)
    valid_durations = [d for d in durations if d > 0]
    if not valid_durations:
        return 0.5

    avg_tenure = sum(valid_durations) / len(valid_durations)

    if avg_tenure < TENURE_VERY_LOW:
        return 0.1
    if avg_tenure < TENURE_LOW:
        return 0.5
    if avg_tenure <= TENURE_HIGH:
        return 1.0
    return 0.8  # > 36 months


def _compute_industry_match_score(
    career_history: list[dict],
    jd_industry: str,
) -> float:
    """
    Check how well the candidate's industry experience aligns with JD.

    Current job matches    → 1.0
    Any past job matches   → 0.6
    No match at all        → 0.2
    """
    if not jd_industry or not career_history:
        return 0.2

    jd_industry_lower = jd_industry.lower().strip()

    for role in career_history:
        industry: str = (role.get("industry", "") or "").lower().strip()
        # Partial match: "AI/ML" in "AI/ML Services" or vice-versa
        if jd_industry_lower in industry or industry in jd_industry_lower:
            if role.get("is_current", False):
                return 1.0
            # Mark a past-match but keep looking for a current match
            # (return 0.6 only if no current match found)

    # Second pass: collect any match (not necessarily current)
    for role in career_history:
        industry = (role.get("industry", "") or "").lower().strip()
        if jd_industry_lower in industry or industry in jd_industry_lower:
            return 0.6

    return 0.2


def _compute_education_score(education: list[dict]) -> float:
    """
    Return the best (highest) institution-tier score from all degrees.
    Defaults to 0.25 if no education records exist.
    """
    if not education:
        return 0.25

    best_score = 0.25
    for edu in education:
        tier: str = (edu.get("tier", "") or "").lower().strip()
        score = TIER_MAP.get(tier, 0.25)
        if score > best_score:
            best_score = score

    return best_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_career_score(
    candidate: dict,
    jd_min_experience: float,
    jd_industry: str,
) -> float:
    """
    Compute the normalised career score (0.0 – 1.0) for a candidate.

    Parameters
    ----------
    candidate          : Full candidate dict (as loaded from candidates.jsonl).
    jd_min_experience  : Minimum years of experience required by the JD.
    jd_industry        : Industry string from the JD (e.g. "AI/ML").

    Returns
    -------
    float in [0.0, 1.0]
    """
    profile: dict        = candidate.get("profile", {}) or {}
    career_history: list = candidate.get("career_history", []) or []
    education: list      = candidate.get("education", []) or []

    years_of_experience: float = float(profile.get("years_of_experience", 0) or 0)

    # ------------------------------------------------------------------
    # 1. experience_score  (30 %)
    # ------------------------------------------------------------------
    experience_score: float = _compute_experience_score(
        years_of_experience, jd_min_experience
    )

    # ------------------------------------------------------------------
    # 2. progression_score  (25 %)
    # ------------------------------------------------------------------
    progression_score: float = _compute_progression_score(career_history)

    # ------------------------------------------------------------------
    # 3. tenure_score  (20 %)
    # ------------------------------------------------------------------
    tenure_score: float = _compute_tenure_score(career_history)

    # ------------------------------------------------------------------
    # 4. industry_match_score  (15 %)
    # ------------------------------------------------------------------
    industry_match_score: float = _compute_industry_match_score(
        career_history, jd_industry
    )

    # ------------------------------------------------------------------
    # 5. education_score  (10 %)
    # ------------------------------------------------------------------
    education_score: float = _compute_education_score(education)

    # ------------------------------------------------------------------
    # Weighted combination → career_score
    # ------------------------------------------------------------------
    career_score: float = (
        WEIGHT_EXPERIENCE  * experience_score
        + WEIGHT_PROGRESSION * progression_score
        + WEIGHT_TENURE      * tenure_score
        + WEIGHT_INDUSTRY    * industry_match_score
        + WEIGHT_EDUCATION   * education_score
    )

    # Clamp to [0.0, 1.0]
    return float(max(0.0, min(1.0, career_score)))
