from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
@dataclass(frozen=True)
class ScoringWeights:
    semantic: float = 0.45
    skill: float = 0.20
    career: float = 0.15
    signal: float = 0.20

    def __post_init__(self) -> None:
        total = self.semantic + self.skill + self.career + self.signal
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.4f}"
            )
@dataclass(frozen=True)
class PenaltyConfig:

    honeypot_multiplier: float = 0.0
    not_open_to_work_multiplier: float = 0.6
    inactive_multiplier: float = 0.7
    low_interview_rate_multiplier: float = 0.8

    inactive_days_threshold: int = 180
    interview_completion_rate_threshold: float = 0.3
@dataclass(frozen=True)
class RankingParams:
    top_k: int = 100
    sort_ascending: bool = False           
    tiebreaker_column: str = "candidate_id"
    tiebreaker_ascending: bool = True      
@dataclass(frozen=True)
class ColumnNames:
    candidate_id: str = "candidate_id"
    semantic_score: str = "semantic_score"
    skill_score: str = "skill_score"
    career_score: str = "career_score"
    signal_score: str = "signal_score"
    is_honeypot: str = "is_honeypot"
    open_to_work: str = "open_to_work"
    days_inactive: str = "days_inactive"
    interview_completion_rate: str = "interview_completion_rate"
    skills: str = "skills"
    years_experience: str = "years_experience"
    github_activity: str = "github_activity"
    current_title: str = "current_title"
    education_level: str = "education_level"
    responsiveness: str = "responsiveness"
    final_score: str = "final_score"
    rank: str = "rank"
    reasoning: str = "reasoning"

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class OutputConfig:
    submissions_dir: Path = _PROJECT_ROOT / "data" / "submissions"
    submission_filename: str = "submission.csv"

    @property
    def submission_path(self) -> Path:
        return self.submissions_dir / self.submission_filename

@dataclass(frozen=True)
class ReasoningConfig:
    high_semantic_threshold: float = 0.75
    high_skill_threshold: float = 0.70
    high_career_threshold: float = 0.70
    high_signal_threshold: float = 0.70

    strong_experience_years: int = 5
    high_github_threshold: float = 0.6
    high_responsiveness_threshold: float = 0.7

    max_skills_in_summary: int = 4
@dataclass(frozen=True)
class RankingEngineConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    penalties: PenaltyConfig = field(default_factory=PenaltyConfig)
    ranking: RankingParams = field(default_factory=RankingParams)
    columns: ColumnNames = field(default_factory=ColumnNames)
    output: OutputConfig = field(default_factory=OutputConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)

DEFAULT_CONFIG: Final[RankingEngineConfig] = RankingEngineConfig()
# ==========================================================
# Embeddings & Retrieval Configuration
# ==========================================================

PROJECT_ROOT = _PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset paths
CANDIDATES_PATH = RAW_DATA_DIR / "candidates.jsonl"
JD_PATH = RAW_DATA_DIR / "job_description.docx"

# Model artifacts
EMBEDDINGS_PATH = MODELS_DIR / "candidate_embeddings.npy"
IDS_PATH = MODELS_DIR / "candidate_ids.pkl"
FAISS_INDEX_PATH = MODELS_DIR / "candidate_index.faiss"

# Embedding configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
TOP_K = 100

# Google Drive dataset
GOOGLE_DRIVE_FILE_ID = "1tB7Otd2EGldaDRu62cVnIoW6_G8A_HaT"