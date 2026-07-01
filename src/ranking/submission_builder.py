"""
Submission Builder for Redrob AI Ranking Engine.

Generates the final CSV artefact:

    candidate_id, rank, score, reasoning

The builder orchestrates the full pipeline when used end-to-end:
score → rank → reason → write CSV.  It can also be used in isolation
to serialise an already-processed DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.ranking.scorer import CandidateScorer
from src.ranking.ranker import CandidateRanker
from src.ranking.reasoning_generator import ReasoningGenerator
from src.utils.config import RankingEngineConfig, DEFAULT_CONFIG
from src.utils.helpers import validate_required_columns
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class SubmissionBuilder:
    """Produces the competition submission CSV.

    The builder can run the **full pipeline** (score → rank → reason → CSV)
    or just serialise a pre-processed DataFrame.

    Args:
        config: Engine configuration.

    Example::

        builder = SubmissionBuilder()
        builder.build(features_df)              # writes submission.csv
        builder.build(features_df, top_k=50)    # custom top-K
    """

    # Columns in the final submission file (in order).
    _SUBMISSION_COLUMNS: list[str] = [
        "candidate_id",
        "rank",
        "score",
        "reasoning",
    ]

    def __init__(self, config: Optional[RankingEngineConfig] = None) -> None:
        self._cfg = config or DEFAULT_CONFIG
        self._col = self._cfg.columns
        self._out = self._cfg.output

        self._scorer = CandidateScorer(self._cfg)
        self._ranker = CandidateRanker(self._cfg)
        self._reasoner = ReasoningGenerator(self._cfg)

        logger.info(
            "SubmissionBuilder initialised  [output=%s]",
            self._out.submission_path,
        )

    # ── public API ──────────────────────────────────────────────────

    def build(
        self,
        features_df: pd.DataFrame,
        *,
        output_path: Optional[Union[str, Path]] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run the full pipeline and write the submission CSV.

        Args:
            features_df: Raw feature DataFrame with component scores
                and penalty columns.
            output_path: Override for the CSV destination.
            top_k: Override for how many candidates to include.

        Returns:
            The final submission DataFrame (also written to disk).

        Raises:
            KeyError: If required columns are missing from *features_df*.
            ValueError: If *features_df* is empty.
        """
        logger.info("═══ Starting submission build pipeline ═══")

        # Step 1 — Score
        scored_df = self._scorer.score(features_df)

        # Step 2 — Rank
        ranked_df = self._ranker.rank(scored_df, top_k=top_k)

        # Step 3 — Generate reasoning
        reasoned_df = self._reasoner.generate(ranked_df)

        # Step 4 — Format and write
        submission_df = self._format_submission(reasoned_df)
        dest = Path(output_path) if output_path else self._out.submission_path
        self._write_csv(submission_df, dest)

        logger.info("═══ Submission build complete ═══")
        return submission_df

    def write_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Serialise an already-processed DataFrame to CSV.

        Use this when you've already run scoring, ranking, and reasoning
        externally and just need to write the file.

        Args:
            df: DataFrame containing at least ``candidate_id``,
                ``final_score`` (or ``score``), ``rank``, and
                ``reasoning``.
            output_path: Override for the CSV destination.

        Returns:
            The path the CSV was written to.
        """
        submission_df = self._format_submission(df)
        dest = Path(output_path) if output_path else self._out.submission_path
        self._write_csv(submission_df, dest)
        return dest

    # ── internals ───────────────────────────────────────────────────

    def _format_submission(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and rename columns to match the submission schema.

        Maps internal column names to the required output columns:
            candidate_id  →  candidate_id
            rank          →  rank
            final_score   →  score
            reasoning     →  reasoning
        """
        out = pd.DataFrame()
        out["candidate_id"] = df[self._col.candidate_id]
        out["rank"] = df[self._col.rank].astype(int)
        out["score"] = df[self._col.final_score].round(6)
        out["reasoning"] = df[self._col.reasoning]
        return out

    @staticmethod
    def _write_csv(df: pd.DataFrame, path: Path) -> None:
        """Write *df* to *path* as a UTF-8 CSV with header."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        size_kb = path.stat().st_size / 1024
        logger.info(
            "Submission written → %s  (%d rows, %.1f KB)",
            path,
            len(df),
            size_kb,
        )
