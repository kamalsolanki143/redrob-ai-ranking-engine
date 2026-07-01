"""
Candidate Scorer for Redrob AI Ranking Engine.

Computes a **final_score** for every candidate row by:

1. Combining four component scores with configurable weights.
2. Applying sequential multiplicative penalties (honeypot, availability,
   inactivity, low-interview-rate).
3. Clipping and normalising the result to [0, 1].

The scorer is stateless: all configuration is injected via
:class:`~src.utils.config.RankingEngineConfig`.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import RankingEngineConfig, DEFAULT_CONFIG
from src.utils.helpers import (
    validate_required_columns,
    safe_get_float,
    clip_scores,
)
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class CandidateScorer:
    """Produces a normalised final score for each candidate.

    Args:
        config: Engine-wide configuration.  Falls back to
            :data:`~src.utils.config.DEFAULT_CONFIG` when omitted.

    Example::

        scorer = CandidateScorer()
        scored_df = scorer.score(features_df)
    """

    def __init__(self, config: Optional[RankingEngineConfig] = None) -> None:
        self._cfg = config or DEFAULT_CONFIG
        self._w = self._cfg.weights
        self._p = self._cfg.penalties
        self._col = self._cfg.columns
        logger.info(
            "CandidateScorer initialised  [weights=%s]  [penalties=%s]",
            self._w,
            self._p,
        )

    # ── public API ──────────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute final scores for every candidate.

        The input DataFrame is **not** mutated; a copy with the new
        ``final_score`` column appended is returned.

        Args:
            df: Feature DataFrame.  Must contain the four component-
                score columns **and** the penalty-flag columns defined
                in :class:`~src.utils.config.ColumnNames`.

        Returns:
            A new DataFrame with a ``final_score`` column (float64,
            range [0, 1]).

        Raises:
            KeyError: If required columns are missing.
            ValueError: If *df* is empty.
        """
        if df.empty:
            raise ValueError("Cannot score an empty DataFrame.")

        self._validate_input(df)
        result = df.copy()

        logger.info("Scoring %d candidates …", len(result))

        # Step 1 — weighted composite score
        result[self._col.final_score] = self._compute_weighted_score(result)

        # Step 2 — apply penalties
        result[self._col.final_score] = self._apply_penalties(result)

        # Step 3 — clip to [0, 1]
        result[self._col.final_score] = clip_scores(
            result[self._col.final_score],
        )

        self._log_score_stats(result[self._col.final_score])
        return result

    # ── internals ───────────────────────────────────────────────────

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Ensure required columns are present."""
        score_cols = [
            self._col.semantic_score,
            self._col.skill_score,
            self._col.career_score,
            self._col.signal_score,
        ]
        penalty_cols = [
            self._col.is_honeypot,
            self._col.open_to_work,
            self._col.days_inactive,
            self._col.interview_completion_rate,
        ]
        validate_required_columns(
            df,
            score_cols + penalty_cols,
            context="CandidateScorer input",
        )

    def _compute_weighted_score(self, df: pd.DataFrame) -> pd.Series:
        """Return the weighted sum of the four component scores.

        Each component is first coerced to float and NaN-filled so a
        missing upstream feature doesn't crash the pipeline.
        """
        semantic = safe_get_float(df[self._col.semantic_score])
        skill = safe_get_float(df[self._col.skill_score])
        career = safe_get_float(df[self._col.career_score])
        signal = safe_get_float(df[self._col.signal_score])

        weighted: pd.Series = (
            self._w.semantic * semantic
            + self._w.skill * skill
            + self._w.career * career
            + self._w.signal * signal
        )
        return weighted

    def _apply_penalties(self, df: pd.DataFrame) -> pd.Series:
        """Apply multiplicative penalties to ``final_score`` in-place order.

        Penalty rules (applied sequentially):
            1. **Honeypot** → score = 0
            2. **Not open to work** → score *= 0.6
            3. **Inactive > threshold** → score *= 0.7
            4. **Low interview completion** → score *= 0.8
        """
        score = df[self._col.final_score].copy()

        # 1 — honeypot
        is_honeypot = df[self._col.is_honeypot].fillna(False).astype(bool)
        n_hp = is_honeypot.sum()
        if n_hp > 0:
            logger.warning("Honeypot detected for %d candidate(s) — zeroing scores.", n_hp)
        score = score.where(~is_honeypot, other=0.0)

        # 2 — not open to work
        open_to_work = df[self._col.open_to_work].fillna(True).astype(bool)
        not_open = ~open_to_work
        n_no = not_open.sum()
        if n_no > 0:
            logger.info(
                "Applying not-open-to-work penalty (×%.2f) to %d candidate(s).",
                self._p.not_open_to_work_multiplier,
                n_no,
            )
        score = score.where(
            ~not_open,
            other=score * self._p.not_open_to_work_multiplier,
        )

        # 3 — inactive too long
        days_inactive = safe_get_float(df[self._col.days_inactive], default=0.0)
        inactive_mask = days_inactive > self._p.inactive_days_threshold
        n_in = inactive_mask.sum()
        if n_in > 0:
            logger.info(
                "Applying inactivity penalty (×%.2f) to %d candidate(s) "
                "(>%d days inactive).",
                self._p.inactive_multiplier,
                n_in,
                self._p.inactive_days_threshold,
            )
        score = score.where(
            ~inactive_mask,
            other=score * self._p.inactive_multiplier,
        )

        # 4 — low interview completion rate
        icr = safe_get_float(
            df[self._col.interview_completion_rate], default=1.0,
        )
        low_icr_mask = icr < self._p.interview_completion_rate_threshold
        n_li = low_icr_mask.sum()
        if n_li > 0:
            logger.info(
                "Applying low-interview-rate penalty (×%.2f) to %d candidate(s) "
                "(rate < %.2f).",
                self._p.low_interview_rate_multiplier,
                n_li,
                self._p.interview_completion_rate_threshold,
            )
        score = score.where(
            ~low_icr_mask,
            other=score * self._p.low_interview_rate_multiplier,
        )

        return score

    @staticmethod
    def _log_score_stats(scores: pd.Series) -> None:
        """Emit summary statistics for the final scores."""
        logger.info(
            "Score stats  →  min=%.4f  max=%.4f  mean=%.4f  median=%.4f  "
            "zeros=%d / %d",
            scores.min(),
            scores.max(),
            scores.mean(),
            scores.median(),
            (scores == 0).sum(),
            len(scores),
        )
