"""
signal_features.py
==================
Computes the `signal_score` (0.0 – 1.0) for a single candidate
based purely on Redrob platform signals (no JD context needed).

Score breakdown (weights sum to 1.0):
  - availability_score    : 35%  open to work + recency + notice period
  - responsiveness_score  : 30%  how quickly & reliably they respond
  - credibility_score     : 20%  verified contact + profile completeness
  - market_demand_score   : 15%  recruiter saves, search appearances, GitHub

Special edge-case handling (as observed in real data):
  - github_activity_score == -1  → treat as 0 (no GitHub linked)
  - offer_acceptance_rate == -1  → treat as 0.5 (no offer history; neutral)

Author : Muskan (Feature Engineering)
"""

from __future__ import annotations

from datetime import date, datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reference "today" for computing days-since-active.
# Use the date the dataset was finalised so results are reproducible.
REFERENCE_DATE: date = date(2026, 6, 11)

# Normalisation ceilings for market-demand signals
MAX_SAVED_BY_RECRUITERS: float = 20.0    # saves/month  → 1.0
MAX_SEARCH_APPEARANCES:  float = 300.0   # appearances/month → 1.0

# Component weights inside signal_score — must sum to 1.0
WEIGHT_AVAILABILITY:   float = 0.35
WEIGHT_RESPONSIVENESS: float = 0.30
WEIGHT_CREDIBILITY:    float = 0.20
WEIGHT_MARKET_DEMAND:  float = 0.15


# ---------------------------------------------------------------------------
# Internal helpers — availability
# ---------------------------------------------------------------------------

def _open_to_work_score(flag: bool | None) -> float:
    """True → 1.0; False / missing → 0.4 (not looking but could be contacted)."""
    return 1.0 if flag else 0.4


def _days_since_active_score(last_active_date_str: str | None) -> float:
    """
    Score how recently the candidate was active on the platform.

    ≤ 30 days   → 1.0
    31–90 days  → 0.8
    91–180 days → 0.5
    > 180 days  → 0.2
    """
    if not last_active_date_str:
        return 0.2  # Unknown → pessimistic

    try:
        last_active = datetime.strptime(last_active_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0.2

    days = (REFERENCE_DATE - last_active).days

    if days <= 30:
        return 1.0
    if days <= 90:
        return 0.8
    if days <= 180:
        return 0.5
    return 0.2


def _notice_period_score(notice_period_days: int | float | None) -> float:
    """
    Candidates who can join faster are more attractive.

    ≤ 30 days   → 1.0
    ≤ 60 days   → 0.8
    ≤ 90 days   → 0.6
    ≤ 120 days  → 0.4
    > 120 days  → 0.2
    """
    days = float(notice_period_days or 0)
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.8
    if days <= 90:
        return 0.6
    if days <= 120:
        return 0.4
    return 0.2


def _compute_availability_score(signals: dict) -> float:
    """
    availability_score = 0.4 × open_to_work
                       + 0.4 × days_since_active_score
                       + 0.2 × notice_period_score
    """
    a = _open_to_work_score(signals.get("open_to_work_flag"))
    b = _days_since_active_score(signals.get("last_active_date"))
    c = _notice_period_score(signals.get("notice_period_days"))
    return 0.4 * a + 0.4 * b + 0.2 * c


# ---------------------------------------------------------------------------
# Internal helpers — responsiveness
# ---------------------------------------------------------------------------

def _response_time_score(avg_hours: float | None) -> float:
    """
    Faster response = higher score.

    ≤ 24 h   → 1.0
    ≤ 72 h   → 0.8
    ≤ 120 h  → 0.6
    ≤ 168 h  → 0.4
    > 168 h  → 0.2
    """
    h = float(avg_hours or 0)
    if h <= 24:
        return 1.0
    if h <= 72:
        return 0.8
    if h <= 120:
        return 0.6
    if h <= 168:
        return 0.4
    return 0.2


def _compute_responsiveness_score(signals: dict) -> float:
    """
    responsiveness_score = 0.4 × recruiter_response_rate
                         + 0.3 × response_time_score
                         + 0.3 × interview_completion_rate
    """
    response_rate: float = float(signals.get("recruiter_response_rate", 0) or 0)
    response_rate = max(0.0, min(1.0, response_rate))

    rt_score: float = _response_time_score(signals.get("avg_response_time_hours"))

    interview_rate: float = float(signals.get("interview_completion_rate", 0) or 0)
    interview_rate = max(0.0, min(1.0, interview_rate))

    return 0.4 * response_rate + 0.3 * rt_score + 0.3 * interview_rate


# ---------------------------------------------------------------------------
# Internal helpers — credibility
# ---------------------------------------------------------------------------

def _offer_acceptance_score(raw_value: float | None) -> float:
    """
    -1 → no offer history → neutral 0.5
    Any other value → clamp to [0.0, 1.0] and use directly
    """
    if raw_value is None:
        return 0.5
    v = float(raw_value)
    if v < 0:  # -1 sentinel
        return 0.5
    return max(0.0, min(1.0, v))


def _compute_credibility_score(signals: dict) -> float:
    """
    credibility_score = 0.25 × verified_email
                      + 0.25 × verified_phone
                      + 0.30 × profile_completeness_score / 100
                      + 0.20 × offer_acceptance_score
    """
    verified_email: float  = 1.0 if signals.get("verified_email") else 0.0
    verified_phone: float  = 1.0 if signals.get("verified_phone") else 0.0
    completeness: float    = float(signals.get("profile_completeness_score", 0) or 0) / 100.0
    completeness           = max(0.0, min(1.0, completeness))
    offer_acc: float       = _offer_acceptance_score(signals.get("offer_acceptance_rate"))

    return (
        0.25 * verified_email
        + 0.25 * verified_phone
        + 0.30 * completeness
        + 0.20 * offer_acc
    )


# ---------------------------------------------------------------------------
# Internal helpers — market demand
# ---------------------------------------------------------------------------

def _github_score(raw_value: float | None) -> float:
    """
    -1 → not a coder / no GitHub → treat as 0 (not penalised, not rewarded).
    0–100 → normalise to [0, 1].
    """
    if raw_value is None:
        return 0.0
    v = float(raw_value)
    if v < 0:
        return 0.0
    return max(0.0, min(1.0, v / 100.0))


def _compute_market_demand_score(signals: dict) -> float:
    """
    market_demand_score = 0.4 × saved_by_recruiters_30d  (norm to 20)
                        + 0.3 × search_appearance_30d    (norm to 300)
                        + 0.3 × github_activity_score    (norm to 100)
    """
    saved: float = min(
        float(signals.get("saved_by_recruiters_30d", 0) or 0) / MAX_SAVED_BY_RECRUITERS,
        1.0,
    )
    search: float = min(
        float(signals.get("search_appearance_30d", 0) or 0) / MAX_SEARCH_APPEARANCES,
        1.0,
    )
    github: float = _github_score(signals.get("github_activity_score"))

    return 0.4 * saved + 0.3 * search + 0.3 * github


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_signal_score(candidate: dict) -> float:
    """
    Compute the normalised signal score (0.0 – 1.0) for a candidate.

    Parameters
    ----------
    candidate : Full candidate dict (as loaded from candidates.jsonl).

    Returns
    -------
    float in [0.0, 1.0]
    """
    signals: dict = candidate.get("redrob_signals", {}) or {}

    # ------------------------------------------------------------------
    # 1. availability_score  (35 %)
    # ------------------------------------------------------------------
    availability_score: float = _compute_availability_score(signals)

    # ------------------------------------------------------------------
    # 2. responsiveness_score  (30 %)
    # ------------------------------------------------------------------
    responsiveness_score: float = _compute_responsiveness_score(signals)

    # ------------------------------------------------------------------
    # 3. credibility_score  (20 %)
    # ------------------------------------------------------------------
    credibility_score: float = _compute_credibility_score(signals)

    # ------------------------------------------------------------------
    # 4. market_demand_score  (15 %)
    # ------------------------------------------------------------------
    market_demand_score: float = _compute_market_demand_score(signals)

    # ------------------------------------------------------------------
    # Weighted combination → signal_score
    # ------------------------------------------------------------------
    signal_score: float = (
        WEIGHT_AVAILABILITY   * availability_score
        + WEIGHT_RESPONSIVENESS * responsiveness_score
        + WEIGHT_CREDIBILITY    * credibility_score
        + WEIGHT_MARKET_DEMAND  * market_demand_score
    )

    # Clamp to [0.0, 1.0]
    return float(max(0.0, min(1.0, signal_score)))
