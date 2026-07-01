from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    *,
    context: str = "",
) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        ctx = f" ({context})" if context else ""
        raise KeyError(
            f"DataFrame{ctx} is missing required columns: {missing}"
        )


def safe_get_float(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(default).astype(np.float64)


def clip_scores(
    series: pd.Series,
    lower: float = 0.0,
    upper: float = 1.0,
) -> pd.Series:
   return series.clip(lower=lower, upper=upper)


def normalise_min_max(
    series: pd.Series,
    *,
    eps: float = 1e-10,
) -> pd.Series:
    s_min = series.min()
    s_max = series.max()
    denom = s_max - s_min
    if denom < eps:
        logger.debug(
            "Min-max range < eps (%.2e); returning zeros.", eps,
        )
        return pd.Series(0.0, index=series.index, dtype=np.float64)
    return (series - s_min) / denom
