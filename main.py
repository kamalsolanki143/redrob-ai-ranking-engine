"""
Redrob AI Ranking Engine — Main Entry Point.

Usage::

    python main.py                          # uses default paths
    python main.py --input data/processed/features.csv
    python main.py --input features.parquet --top-k 50 --output results.csv
    python main.py --demo                   # run with synthetic data

The script orchestrates the full pipeline:
    feature data → score → rank → reason → submission.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so ``src.*`` imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ranking.submission_builder import SubmissionBuilder
from src.utils.config import DEFAULT_CONFIG, RankingEngineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)


# ──────────────────────────────────────────────────────────────────────
# Synthetic demo data (for quick validation)
# ──────────────────────────────────────────────────────────────────────

def _generate_demo_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic feature DataFrame for demonstration.

    Args:
        n: Number of candidate rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame matching the expected feature schema.
    """
    rng = np.random.default_rng(seed)
    col = DEFAULT_CONFIG.columns

    skills_pool = [
        "Python, Machine Learning, TensorFlow, NLP",
        "Java, Spring Boot, Microservices, AWS",
        "React, TypeScript, Node.js, GraphQL",
        "Python, Data Engineering, Spark, Airflow, SQL",
        "Go, Kubernetes, Docker, CI/CD",
        "Python, Deep Learning, PyTorch, Computer Vision",
        "Scala, Kafka, Flink, Data Streaming",
        "Python, FastAPI, PostgreSQL, Redis",
        "JavaScript, Vue.js, CSS, Figma, UX Design",
        "Rust, Systems Programming, WebAssembly",
    ]

    titles_pool = [
        "Senior ML Engineer",
        "Backend Developer",
        "Full-Stack Developer",
        "Data Engineer",
        "DevOps Engineer",
        "Research Scientist",
        "Staff Engineer",
        "Lead Developer",
        "Software Architect",
        "ML Platform Engineer",
    ]

    education_pool = [
        "M.S. Computer Science",
        "B.Tech Computer Science",
        "Ph.D. Machine Learning",
        "B.S. Software Engineering",
        "M.S. Data Science",
        "B.S. Mathematics",
    ]

    df = pd.DataFrame({
        col.candidate_id: [f"CAND-{i:05d}" for i in range(1, n + 1)],
        col.semantic_score: rng.beta(5, 2, size=n),
        col.skill_score: rng.beta(4, 3, size=n),
        col.career_score: rng.beta(3, 2, size=n),
        col.signal_score: rng.beta(4, 2, size=n),
        col.is_honeypot: rng.choice([True, False], size=n, p=[0.02, 0.98]),
        col.open_to_work: rng.choice([True, False], size=n, p=[0.75, 0.25]),
        col.days_inactive: rng.choice(
            [0, 30, 90, 150, 200, 365], size=n, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05],
        ),
        col.interview_completion_rate: rng.beta(8, 2, size=n),
        col.skills: rng.choice(skills_pool, size=n),
        col.years_experience: rng.integers(1, 20, size=n),
        col.github_activity: rng.beta(3, 4, size=n),
        col.current_title: rng.choice(titles_pool, size=n),
        col.education_level: rng.choice(education_pool, size=n),
        col.responsiveness: rng.beta(5, 2, size=n),
    })

    # Sprinkle some NaN to test robustness
    for sparse_col in [col.github_activity, col.responsiveness, col.years_experience]:
        mask = rng.random(size=n) < 0.05
        df.loc[mask, sparse_col] = np.nan

    logger.info("Generated %d synthetic candidates for demo.", n)
    return df


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redrob AI Ranking Engine — Candidate Ranking Pipeline",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to feature CSV/Parquet. Omit to use --demo.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path for the output submission CSV.  "
             "Default: data/submissions/submission.csv",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="Number of top candidates to include (default: 100).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic demo data (500 candidates).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the ranking pipeline."""
    args = _parse_args()

    if args.demo:
        logger.info("Running in demo mode.")
        features_df = _generate_demo_data()
    else:
        default_file = Path("data/processed/feature_scores.parquet")

        if args.input is None:
            args.input = default_file
            logger.info(f"Using feature file: {args.input}")
        else:
            args.input = Path(args.input)

        if not args.input.exists():
            raise FileNotFoundError(
                f"Feature file not found: {args.input}\n"
                "Run:\n"
                "python -m src.features.feature_combiner"
            )

        if args.input.suffix == ".parquet":
            features_df = pd.read_parquet(args.input)
        else:
            features_df = pd.read_csv(args.input)

    # ── build submission ──
    builder = SubmissionBuilder()
    submission = builder.build(
        features_df,
        output_path=args.output,
        top_k=args.top_k,
    )

    # ── preview ──
    print("\n" + "=" * 72)
    print("  SUBMISSION PREVIEW (top 10)")
    print("=" * 72)
    preview = submission.head(10).to_string(index=False)
    print(preview)
    print("=" * 72)
    print(f"  Total candidates in submission: {len(submission)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
