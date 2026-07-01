"""
Deterministic Reasoning Generator for Redrob AI Ranking Engine.

Produces **recruiter-friendly, human-readable explanations** for why
each candidate was ranked where they are.  All reasoning is built from
template logic — **no LLM API calls**.

The generator inspects each candidate's score components, experience,
skills, and behavioural signals, then assembles a concise narrative
paragraph from matching template fragments.

Example output::

    "Strong Python and ML background, 6 years experience, high GitHub
     activity, open to work and highly responsive."
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import RankingEngineConfig, DEFAULT_CONFIG
from src.utils.helpers import safe_get_float
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class ReasoningGenerator:
    """Generates deterministic, template-based reasoning strings.

    Args:
        config: Engine configuration.

    Example::

        gen = ReasoningGenerator()
        df_with_reasoning = gen.generate(ranked_df)
    """

    def __init__(self, config: Optional[RankingEngineConfig] = None) -> None:
        self._cfg = config or DEFAULT_CONFIG
        self._col = self._cfg.columns
        self._rc = self._cfg.reasoning
        logger.info("ReasoningGenerator initialised.")

    # ── public API ──────────────────────────────────────────────────

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append a ``reasoning`` column to *df*.

        Args:
            df: Ranked/scored DataFrame.

        Returns:
            A new DataFrame with the ``reasoning`` column populated.
        """
        if df.empty:
            logger.warning("Empty DataFrame — skipping reasoning generation.")
            result = df.copy()
            result[self._col.reasoning] = pd.Series(dtype="object")
            return result

        result = df.copy()
        logger.info("Generating reasoning for %d candidates …", len(result))

        result[self._col.reasoning] = result.apply(
            self._build_reasoning_for_row, axis=1,
        )

        blank_count = result[self._col.reasoning].eq("").sum()
        if blank_count > 0:
            logger.warning(
                "%d candidate(s) received empty reasoning strings.", blank_count,
            )

        logger.info("Reasoning generation complete.")
        return result

    # ── per-row logic ───────────────────────────────────────────────

    def _build_reasoning_for_row(self, row: pd.Series) -> str:
        """Assemble reasoning fragments for a single candidate row.

        The method is intentionally tolerant of missing columns —
        fragments are only appended when the relevant data exists.
        """
        fragments: list[str] = []

        # ── skill background ───────────────────────────────────────
        fragments.extend(self._skill_fragments(row))

        # ── experience ─────────────────────────────────────────────
        fragments.extend(self._experience_fragments(row))

        # ── education ──────────────────────────────────────────────
        fragments.extend(self._education_fragments(row))

        # ── score-component insights ───────────────────────────────
        fragments.extend(self._score_component_fragments(row))

        # ── behavioural signals ────────────────────────────────────
        fragments.extend(self._signal_fragments(row))

        # ── penalty disclosures ────────────────────────────────────
        fragments.extend(self._penalty_fragments(row))

        reasoning = ", ".join(fragments) + "." if fragments else ""
        # Capitalise the first letter
        if reasoning:
            reasoning = reasoning[0].upper() + reasoning[1:]
        return reasoning

    # ── fragment builders ───────────────────────────────────────────

    def _skill_fragments(self, row: pd.Series) -> list[str]:
        """Produce fragments describing the candidate's skill set."""
        frags: list[str] = []
        skills_raw = self._safe_value(row, self._col.skills)

        if skills_raw is not None:
            # `skills` may be a list, comma-separated string, or similar
            if isinstance(skills_raw, str):
                skills_list = [
                    s.strip() for s in skills_raw.split(",") if s.strip()
                ]
            elif isinstance(skills_raw, (list, tuple)):
                skills_list = [str(s).strip() for s in skills_raw if s]
            else:
                skills_list = []

            if skills_list:
                displayed = skills_list[: self._rc.max_skills_in_summary]
                skill_str = ", ".join(displayed)
                if len(skills_list) > self._rc.max_skills_in_summary:
                    extra = len(skills_list) - self._rc.max_skills_in_summary
                    frags.append(
                        f"strong {skill_str} (+{extra} more) background"
                    )
                else:
                    frags.append(f"strong {skill_str} background")

        return frags

    def _experience_fragments(self, row: pd.Series) -> list[str]:
        """Produce fragments about work experience duration."""
        frags: list[str] = []
        years = self._safe_numeric(row, self._col.years_experience)

        if years is not None:
            y = int(years)
            if y >= self._rc.strong_experience_years:
                frags.append(f"{y} years experience")
            elif y > 0:
                frags.append(f"{y} years experience (early career)")

        title = self._safe_value(row, self._col.current_title)
        if title and isinstance(title, str) and title.strip():
            frags.append(f"currently {title.strip()}")

        return frags

    def _education_fragments(self, row: pd.Series) -> list[str]:
        """Produce fragments about education level."""
        frags: list[str] = []
        edu = self._safe_value(row, self._col.education_level)

        if edu and isinstance(edu, str) and edu.strip():
            frags.append(f"{edu.strip()} education")

        return frags

    def _score_component_fragments(self, row: pd.Series) -> list[str]:
        """Highlight notably strong component scores."""
        frags: list[str] = []

        semantic = self._safe_numeric(row, self._col.semantic_score)
        if semantic is not None and semantic >= self._rc.high_semantic_threshold:
            frags.append("excellent profile-to-job semantic match")

        skill = self._safe_numeric(row, self._col.skill_score)
        if skill is not None and skill >= self._rc.high_skill_threshold:
            frags.append("highly relevant skill set")

        career = self._safe_numeric(row, self._col.career_score)
        if career is not None and career >= self._rc.high_career_threshold:
            frags.append("strong career progression")

        signal = self._safe_numeric(row, self._col.signal_score)
        if signal is not None and signal >= self._rc.high_signal_threshold:
            frags.append("positive engagement signals")

        return frags

    def _signal_fragments(self, row: pd.Series) -> list[str]:
        """Describe behavioural / platform signals."""
        frags: list[str] = []

        # GitHub activity
        gh = self._safe_numeric(row, self._col.github_activity)
        if gh is not None and gh >= self._rc.high_github_threshold:
            frags.append("high GitHub activity")

        # Open-to-work
        otw = self._safe_value(row, self._col.open_to_work)
        if otw is not None:
            if self._coerce_bool(otw):
                frags.append("open to work")
            else:
                frags.append("not currently marked as open to work")

        # Responsiveness
        resp = self._safe_numeric(row, self._col.responsiveness)
        if resp is not None and resp >= self._rc.high_responsiveness_threshold:
            frags.append("highly responsive")

        return frags

    def _penalty_fragments(self, row: pd.Series) -> list[str]:
        """Transparently note any penalties that were applied."""
        frags: list[str] = []
        penalties = self._cfg.penalties

        # Honeypot
        hp = self._safe_value(row, self._col.is_honeypot)
        if hp is not None and self._coerce_bool(hp):
            frags.append("⚠ flagged as honeypot (score zeroed)")

        # Inactivity
        days = self._safe_numeric(row, self._col.days_inactive)
        if days is not None and days > penalties.inactive_days_threshold:
            frags.append(
                f"inactive for {int(days)} days (penalty applied)"
            )

        # Interview completion rate
        icr = self._safe_numeric(row, self._col.interview_completion_rate)
        if (
            icr is not None
            and icr < penalties.interview_completion_rate_threshold
        ):
            frags.append(
                f"low interview completion rate ({icr:.0%}, penalty applied)"
            )

        return frags

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _safe_value(row: pd.Series, col: str) -> Any | None:
        """Return the value if the column exists and is not NaN/None."""
        if col not in row.index:
            return None
        val = row[col]
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        return val

    @staticmethod
    def _safe_numeric(row: pd.Series, col: str) -> float | None:
        """Return a float if the column exists and is numeric."""
        if col not in row.index:
            return None
        val = row[col]
        try:
            f = float(val)
            return None if np.isnan(f) or np.isinf(f) else f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """Best-effort coercion to bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)
