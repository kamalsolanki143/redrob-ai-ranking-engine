"""
Candidate Ranker for Redrob AI Ranking Engine.

Responsibilities:
- Accept a scored DataFrame containing final_score.
- Sort candidates deterministically.
- Assign ranks.
- Select top-K candidates.
- Return ranked DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..utils.config import RankingEngineConfig, DEFAULT_CONFIG
from ..utils.helpers import validate_required_columns
from ..utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class CandidateRanker:
    """
    Sorts scored candidates and selects the top-K.
    """

    def __init__(
        self,
        config: Optional[RankingEngineConfig] = None,
    ) -> None:
        self._cfg = config or DEFAULT_CONFIG
        self._rp = self._cfg.ranking
        self._col = self._cfg.columns

        logger.info(
            "CandidateRanker initialized [top_k=%d]",
            self._rp.top_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        df: pd.DataFrame,
        *,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rank candidates using final_score.

        Args:
            df:
                DataFrame containing candidate_id and final_score.
            top_k:
                Override default top-k value.

        Returns:
            Ranked DataFrame.
        """
        if df.empty:
            raise ValueError("Cannot rank an empty DataFrame.")

        self._validate_input(df)

        effective_k = (
            top_k if top_k is not None else self._rp.top_k
        )

        result = df.copy()

        # --------------------------------------------------------------
        # Handle missing scores
        # --------------------------------------------------------------

        if result[self._col.final_score].isna().any():
            logger.warning(
                "Found NaN values in final_score. Replacing with 0.0"
            )

            result[self._col.final_score] = (
                result[self._col.final_score]
                .fillna(0.0)
            )

        # --------------------------------------------------------------
        # Remove duplicate candidate IDs
        # --------------------------------------------------------------

        duplicate_count = (
            result[self._col.candidate_id]
            .duplicated()
            .sum()
        )

        if duplicate_count > 0:
            logger.warning(
                "Found %d duplicate candidate IDs. "
                "Keeping first occurrence.",
                duplicate_count,
            )

            result = result.drop_duplicates(
                subset=[self._col.candidate_id],
                keep="first",
            )

        # --------------------------------------------------------------
        # Sort candidates
        # --------------------------------------------------------------

        logger.info(
            "Sorting %d candidates...",
            len(result),
        )

        result = result.sort_values(
            by=[
                self._col.final_score,
                self._rp.tiebreaker_column,
            ],
            ascending=[
                self._rp.sort_ascending,
                self._rp.tiebreaker_ascending,
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        # --------------------------------------------------------------
        # Assign ranks
        # --------------------------------------------------------------

        result[self._col.rank] = range(
            1,
            len(result) + 1,
        )

        # --------------------------------------------------------------
        # Select Top-K
        # --------------------------------------------------------------

        n_selected = min(
            effective_k,
            len(result),
        )

        ranked = result.head(
            n_selected
        ).copy()

        logger.info(
            "Selected top %d candidates "
            "(requested=%d, available=%d)",
            n_selected,
            effective_k,
            len(result),
        )

        self._log_rank_summary(ranked)

        return ranked

    def load_and_rank(
        self,
        source: Union[str, Path, pd.DataFrame],
        *,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load candidate data and rank it.
        """

        if isinstance(source, pd.DataFrame):
            df = source

        else:
            path = Path(source)

            logger.info(
                "Loading feature data from %s",
                path,
            )

            if path.suffix == ".parquet":
                df = pd.read_parquet(path)

            elif path.suffix == ".csv":
                df = pd.read_csv(path)

            elif path.suffix == ".tsv":
                df = pd.read_csv(
                    path,
                    sep="\t",
                )

            else:
                raise ValueError(
                    f"Unsupported file format: {path.suffix}"
                )

            logger.info(
                "Loaded %d rows × %d columns",
                len(df),
                len(df.columns),
            )

        return self.rank(
            df,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate required columns.
        """

        validate_required_columns(
            df,
            [
                self._col.candidate_id,
                self._col.final_score,
            ],
            context="CandidateRanker input",
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_rank_summary(
        self,
        ranked: pd.DataFrame,
    ) -> None:
        """
        Log top and bottom candidates.
        """

        if ranked.empty:
            return

        cols = [
            self._col.rank,
            self._col.candidate_id,
            self._col.final_score,
        ]

        top5 = ranked.head(5)[cols]

        bottom5 = ranked.tail(5)[cols]

        logger.info(
            "Top 5 candidates:\n%s",
            top5.to_string(index=False),
        )

        logger.info(
            "Bottom 5 candidates (selected set):\n%s",
            bottom5.to_string(index=False),
        )s