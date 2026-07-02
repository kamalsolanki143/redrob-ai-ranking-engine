"""
feature_combiner.py
===================
Main entry point for Muskan's feature-engineering pipeline.

Pipeline:
  1. Parse the job description (JSON or fallback text heuristics).
  2. Stream candidates.jsonl line-by-line and pre-filter to ~500 candidates
     using a fast keyword-hit count (BM25-style, no ML).
  3. Run all three feature scorers on the filtered set.
  4. Save results as data/processed/feature_scores.parquet

Output columns: candidate_id, skill_score, career_score, signal_score

Performance target: ≤ 5 minutes on MacBook Air M-series for 1 lakh records.
Strategy:
  - Streaming reads — never hold the full file in RAM.
  - Pre-filter uses only string ops (no imports of heavy libraries).
  - Scoring is pure Python math — no model inference.

Usage:
  python src/features/feature_combiner.py
  python src/features/feature_combiner.py --jd data/raw/job_description.json
  python src/features/feature_combiner.py --top 500

Author : Muskan (Feature Engineering)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — enable running as a standalone script OR as a module
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent          # .../src
_ROOT = _SRC.parent           # .../

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(1, str(_ROOT))

from features.skill_features  import compute_skill_score
from features.career_features import compute_career_score
from features.signal_features import compute_signal_score

from src.validation import validate_candidate, get_penalty_multiplier


# ---------------------------------------------------------------------------
# Default file paths (relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_CANDIDATES_PATH = _ROOT / "data" / "raw" / "candidates.jsonl"
DEFAULT_JD_PATH         = _ROOT / "data" / "raw" / "job_description.docx"
DEFAULT_OUTPUT_PATH     = _ROOT / "data" / "processed" / "feature_scores.parquet"

# FAISS output files (relative to project root)
FAISS_INDEX_PATH = _ROOT / "models" / "candidate_index.faiss"
EMBEDDINGS_PATH = _ROOT / "models" / "candidate_embeddings.npy"
IDS_PATH = _ROOT / "models" / "candidate_ids.pkl"

# How many top candidates to keep after pre-filter
DEFAULT_TOP_N = 500

# Minimum number of keyword hits for a candidate to survive the pre-filter
MIN_HITS_THRESHOLD = 1

# ===========================================================================
# FAISS availability check
# ===========================================================================

def faiss_index_available() -> bool:
    """Return True if Krish's FAISS index + embedding files all exist on disk."""
    return FAISS_INDEX_PATH.exists() and EMBEDDINGS_PATH.exists() and IDS_PATH.exists()


# ===========================================================================
# FAISS-based retrieval  (primary mode — uses Krish's pipeline)
# ===========================================================================

def faiss_retrieve_candidates(
    jd_path: Path,
    candidates_jsonl: Path,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    """
    Use Krish's FAISS semantic search to retrieve top-N candidates.

    Calls retrieve_top_candidates() from src/embeddings/similarity.py,
    which returns a DataFrame of (candidate_id, semantic_score).
    Then fetches full candidate dicts from candidates.jsonl by ID.

    Requires Krish's output files:
      - models/faiss_index.bin
      - data/processed/candidate_embeddings.npy
      - data/processed/candidate_ids.pkl
    """
    print("[INFO] FAISS mode: using Krish's semantic search pipeline…")
    t0 = time.time()

    # Lazily import Krish's similarity module (only when FAISS mode active)
    try:
        from src.embeddings.similarity import retrieve_top_candidates
    except ImportError as e:
        raise ImportError(
            f"Cannot import similarity module. Original error: {e}"
        )

    # FAISS retrieval → DataFrame[candidate_id, semantic_score]
    faiss_df = retrieve_top_candidates(
        jd_path=jd_path,
        embeddings_path=EMBEDDINGS_PATH,
        ids_path=IDS_PATH,
        index_path=FAISS_INDEX_PATH,
        top_k=top_n,
    )

    top_ids: set[str] = set(faiss_df["candidate_id"].tolist())
    print(f"[INFO] FAISS returned {len(top_ids)} candidate IDs. Fetching full profiles…")

    # Fetch full candidate dicts from candidates.jsonl by ID
    id_to_candidate: dict[str, dict] = {}

    for candidate in stream_candidates(candidates_jsonl):
        cid = candidate.get("candidate_id", "")
        if cid in top_ids:
            id_to_candidate[cid] = candidate

        if len(id_to_candidate) == len(top_ids):
            break  # found all — stop early

    # Create candidate_id -> semantic_score mapping
    score_map = dict(
        zip(
            faiss_df["candidate_id"],
            faiss_df["semantic_score"],
        )
    )

    # Return candidates in FAISS order and attach semantic score
    ordered = []
    for cid in faiss_df["candidate_id"]:
        if cid in id_to_candidate:
            candidate = id_to_candidate[cid]
            candidate["semantic_score"] = float(score_map[cid])
            ordered.append(candidate)

    elapsed = time.time() - t0
    print(f"[INFO] FAISS retrieval done: {len(ordered)} candidates in {elapsed:.1f}s.")
    return ordered


# ===========================================================================

def load_jd_from_json(jd_path: Path) -> dict:
    """
    Load a structured job description from a JSON file.

    Expected structure:
    {
        "title": "...",
        "required_skills": [...],
        "preferred_skills": [...],
        "min_experience_years": 3.0,
        "industry": "AI/ML",
        "description": "..."
    }
    """
    with open(jd_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate / normalise with safe defaults
    jd: dict = {
        "title":               data.get("title", ""),
        "required_skills":     data.get("required_skills", []),
        "preferred_skills":    data.get("preferred_skills", []),
        "min_experience_years": float(data.get("min_experience_years", 0) or 0),
        "industry":            data.get("industry", ""),
        "description":         data.get("description", ""),
    }
    return jd


def parse_jd_from_text(jd_text: str) -> dict:
    """
    Fallback: extract JD fields from raw text using simple heuristics.
    No API calls, no ML.  Works well-enough for common JD formats.

    Returns a dict with the same structure as load_jd_from_json.
    """
    jd_text = jd_text or ""

    # --- Title -----------------------------------------------------------
    title = ""
    title_match = re.search(
        r"(?:job title|position|role)\s*[:\-]\s*(.+)",
        jd_text, re.IGNORECASE
    )
    if title_match:
        title = title_match.group(1).strip()

    # --- Min experience --------------------------------------------------
    min_exp = 0.0
    exp_match = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of)?\s*experience",
        jd_text, re.IGNORECASE
    )
    if exp_match:
        min_exp = float(exp_match.group(1))

    # --- Industry --------------------------------------------------------
    industry = ""
    industry_patterns = [
        r"industry\s*[:\-]\s*(.+)",
        r"sector\s*[:\-]\s*(.+)",
    ]
    for pat in industry_patterns:
        m = re.search(pat, jd_text, re.IGNORECASE)
        if m:
            industry = m.group(1).strip().split("\n")[0]
            break

    # --- Skills ----------------------------------------------------------
    # Common skill keywords that appear in tech JDs
    COMMON_SKILLS = [
        "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++",
        "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala",
        "TensorFlow", "PyTorch", "Keras", "scikit-learn", "XGBoost",
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "Spark", "Kafka", "Airflow", "dbt", "Hadoop",
        "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
        "React", "Angular", "Vue", "Node.js", "FastAPI", "Flask", "Django",
        "MLOps", "Data Engineering", "Feature Engineering",
        "Pandas", "NumPy", "Matplotlib", "Seaborn",
        "FAISS", "LangChain", "RAG", "LLM", "Transformers",
        "Git", "CI/CD", "Linux", "REST", "GraphQL",
        "Power BI", "Tableau", "Excel",
        "R", "MATLAB", "Statistics", "A/B Testing",
    ]

    found_skills: list[str] = []
    jd_lower = jd_text.lower()
    for skill in COMMON_SKILLS:
        if skill.lower() in jd_lower:
            found_skills.append(skill)

    # Also look for bullet-list skills after "required skills:" header
    bullet_section = re.search(
        r"required skills?\s*[:\-](.*?)(?:preferred|nice[- ]to[- ]have|responsibilities|$)",
        jd_text, re.IGNORECASE | re.DOTALL
    )
    if bullet_section:
        bullet_text = bullet_section.group(1)
        # Pull out short phrases (1–3 words) from bullet items
        bullets = re.findall(r"[•\-\*]\s*(.+)", bullet_text)
        for b in bullets:
            skill_candidate = b.strip().split("(")[0].strip()  # strip notes
            if 1 <= len(skill_candidate.split()) <= 4:
                if skill_candidate not in found_skills:
                    found_skills.append(skill_candidate)

    return {
        "title":               title,
        "required_skills":     found_skills,
        "preferred_skills":    [],
        "min_experience_years": min_exp,
        "industry":            industry,
        "description":         jd_text,
    }


def load_jd(jd_path: Path) -> dict:
    """
    Smart JD loader: tries JSON first, falls back to text parsing.
    """
    if not jd_path.exists():
        print(f"[WARN] JD file not found at {jd_path}. Using empty JD defaults.")
        return {
            "title": "",
            "required_skills": [],
            "preferred_skills": [],
            "min_experience_years": 0.0,
            "industry": "",
            "description": "",
        }

    suffix = jd_path.suffix.lower()
    if suffix == ".json":
        return load_jd_from_json(jd_path)

    # Plain-text or docx-exported-to-txt fallback
    with open(jd_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return parse_jd_from_text(text)


# ===========================================================================
# Streaming candidate reader
# ===========================================================================

def stream_candidates(jsonl_path: Path) -> Iterator[dict]:
    """
    Yield one candidate dict at a time from a .jsonl file.
    Never loads the full file into memory.
    Skips malformed lines with a warning instead of crashing.
    """
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno}: skipping malformed JSON — {e}")


# ===========================================================================
# BM25-style keyword pre-filter
# ===========================================================================

def _build_candidate_text(candidate: dict) -> str:
    """
    Concatenate all searchable text fields of a candidate into one
    lowercase string for fast keyword matching.
    """
    parts: list[str] = []

    profile = candidate.get("profile", {}) or {}
    parts.append(profile.get("headline", "") or "")
    parts.append(profile.get("summary", "") or "")
    parts.append(profile.get("current_title", "") or "")
    parts.append(profile.get("current_industry", "") or "")

    for skill in candidate.get("skills", []) or []:
        parts.append(skill.get("name", "") or "")

    for role in candidate.get("career_history", []) or []:
        parts.append(role.get("title", "") or "")
        parts.append(role.get("description", "") or "")

    for edu in candidate.get("education", []) or []:
        parts.append(edu.get("field_of_study", "") or "")

    return " ".join(parts).lower()


def _keyword_hit_count(text: str, keywords: list[str]) -> int:
    """Count how many unique keywords appear in the text (case-insensitive)."""
    hits = 0
    for kw in keywords:
        if kw.lower() in text:
            hits += 1
    return hits


def pre_filter_candidates(
    jsonl_path: Path,
    keywords: list[str],
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    """
    Stream through all candidates, score each by keyword hit count, and
    return the top `top_n` candidates ranked by hit count descending.

    Memory strategy:
      - Only stores (hit_count, candidate) pairs.
      - Uses a simple list + sort (fine for 1 lakh records in RAM;
        each compressed candidate dict is ~3 KB → ~300 MB total).
      - For even lower RAM, a heapq.nlargest approach could be used.

    Time complexity: O(N × K) where N=100k, K=len(keywords).
    Typical runtime: < 60 seconds on MacBook Air M-series.
    """
    if not keywords:
        # No keywords → can't filter; stream first top_n records
        print("[WARN] No keywords extracted from JD. Taking first top_n candidates.")
        results: list[dict] = []
        for candidate in stream_candidates(jsonl_path):
            results.append(candidate)
            if len(results) >= top_n:
                break
        return results

    print(f"[INFO] Pre-filter: scanning candidates.jsonl with {len(keywords)} keywords…")
    t0 = time.time()

    scored: list[tuple[int, dict]] = []  # (hit_count, candidate)
    total = 0
    passed = 0

    for candidate in stream_candidates(jsonl_path):
        total += 1
        text   = _build_candidate_text(candidate)
        hits   = _keyword_hit_count(text, keywords)
        if hits >= MIN_HITS_THRESHOLD:
            scored.append((hits, candidate))
            passed += 1

        # Progress ticker every 10k
        if total % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  … {total:>7,} scanned | {passed:>6,} passed | {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    print(
        f"[INFO] Pre-filter done: {total:,} scanned, {passed:,} passed, "
        f"{elapsed:.1f}s elapsed."
    )

    # Sort by hit count descending, take top_n
    scored.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [c for _, c in scored[:top_n]]

    print(f"[INFO] Selected top {len(top_candidates)} candidates for feature scoring.")
    return top_candidates


# ===========================================================================
# Feature scoring
# ===========================================================================

def score_candidates(
    candidates: list[dict],
    jd: dict,
) -> pd.DataFrame:
    """
    Run all three feature scorers on the candidate list and return a DataFrame.

    Output columns: candidate_id, skill_score, career_score, signal_score,
                    is_honeypot, risk_score, validation_tier
    """
    jd_skills     = (jd.get("required_skills", []) or []) + (jd.get("preferred_skills", []) or [])
    jd_skills     = [s for s in jd_skills if s]  # remove empty strings
    jd_min_exp    = float(jd.get("min_experience_years", 0) or 0)
    jd_industry   = str(jd.get("industry", "") or "")

    records: list[dict] = []
    n = len(candidates)

    print(f"[INFO] Scoring {n} candidates…")
    t0 = time.time()

    for i, candidate in enumerate(candidates, start=1):
        cid = candidate.get("candidate_id", f"UNKNOWN_{i}")
        try:
            skill_score = compute_skill_score(candidate, jd_skills)
            career_score = compute_career_score(candidate, jd_min_exp, jd_industry)
            signal_score = compute_signal_score(candidate)

            signals = candidate.get("redrob_signals", {}) or {}

            open_to_work = bool(signals.get("open_to_work_flag", True))

            last_active = signals.get("last_active_date")
            if last_active:
                from datetime import datetime
                reference_date = datetime(2026, 6, 11).date()
                last_date = datetime.strptime(last_active, "%Y-%m-%d").date()
                days_inactive = (reference_date - last_date).days
            else:
                days_inactive = 999

            interview_completion_rate = float(
                signals.get("interview_completion_rate", 1.0) or 1.0
            )

        except Exception as e:
            print(f"[WARN] Error scoring {cid}: {e}. Using 0.0 defaults.")
            skill_score = career_score = signal_score = 0.0

            open_to_work = True
            days_inactive = 999
            interview_completion_rate = 1.0
        # --- Honeypot / Validation Check ---
        # Run validation to detect fake/suspicious profiles.
        # The penalty multiplier is applied downstream by scorer.py/ranker.py:
        #   honeypot   → multiplier = 0.0 (excluded from Top 100)
        #   suspicious → multiplier = 0.7 (penalized)
        #   clean      → multiplier = 1.0 (no penalty)
        try:
            validation_result = validate_candidate(candidate)
            is_honeypot = validation_result.is_honeypot
            risk_score = validation_result.risk_score
            validation_tier = validation_result.tier
        except Exception as e:
            print(f"[WARN] Validation error for {cid}: {e}. Assuming clean.")
            is_honeypot = False
            risk_score = 0
            validation_tier = "clean"

        records.append(
            {
                "candidate_id": cid,
                "semantic_score": round(candidate.get("semantic_score", 0.0), 4),
                "skill_score": round(skill_score, 4),
                "career_score": round(career_score, 4),
                "signal_score": round(signal_score, 4),
                "is_honeypot": is_honeypot,
                "open_to_work": open_to_work,
                "days_inactive": days_inactive,
                "interview_completion_rate": round(interview_completion_rate, 4),
                "risk_score": risk_score,
                "validation_tier": validation_tier,
            }
        )

        if i % 100 == 0:
            print(f"  … {i}/{n} scored")

    elapsed = time.time() - t0
    print(f"[INFO] Scoring done in {elapsed:.1f}s.")

    return pd.DataFrame(records)


# ===========================================================================
# Output
# ===========================================================================

def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Save the feature DataFrame to Parquet format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Saved {len(df)} rows → {output_path}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"\nSample output (first 5 rows):\n{df.head().to_string(index=False)}")


# ===========================================================================
# Main
# ===========================================================================

def run_pipeline(
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    jd_path: Path         = DEFAULT_JD_PATH,
    output_path: Path     = DEFAULT_OUTPUT_PATH,
    top_n: int            = DEFAULT_TOP_N,
    mode: str             = "auto",   # "auto" | "faiss" | "bm25"
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Returns the feature DataFrame (also saved to disk).
    """
    print("=" * 60)
    print("  Redrob AI — Feature Engineering Pipeline (Muskan)")
    print("=" * 60)
    wall_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load JD
    # ------------------------------------------------------------------
    print(f"\n[STEP 1] Loading job description from: {jd_path}")
    jd = load_jd(jd_path)
    all_skills = (jd.get("required_skills", []) or []) + (jd.get("preferred_skills", []) or [])
    print(f"  Title          : {jd.get('title', 'N/A')}")
    print(f"  Required skills: {jd.get('required_skills', [])}")
    print(f"  Preferred skills: {jd.get('preferred_skills', [])}")
    print(f"  Min experience : {jd.get('min_experience_years', 0)} yrs")
    print(f"  Industry       : {jd.get('industry', 'N/A')}")

    # ------------------------------------------------------------------
    # Step 2: Retrieve candidates (FAISS or BM25 Fallback)
    # ------------------------------------------------------------------
    print(f"\n[STEP 2] Retrieving candidates from: {candidates_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"candidates.jsonl not found at {candidates_path}. "
            "Please place it in data/raw/ or update the path."
        )

    use_faiss = (mode == "faiss") or (mode == "auto" and faiss_index_available())

    if mode == "faiss" and not faiss_index_available():
        raise FileNotFoundError(
            "FAISS mode requested but index files are missing.\n"
            f"  Expected: {FAISS_INDEX_PATH}\n"
            f"           {EMBEDDINGS_PATH}\n"
            f"           {IDS_PATH}\n"
            "Ask Krish to run generate_embeddings.py and faiss_index.py first."
        )

    if use_faiss:
        print(f"  [MODE] FAISS semantic search ✅  (Krish's pipeline)")
        top_candidates = faiss_retrieve_candidates(jd_path, candidates_path, top_n)
    else:
        if mode == "auto":
            print(f"  [MODE] BM25 keyword filter ⚡  (FAISS index not yet available — fallback)")
            print(f"         Will switch to FAISS automatically once Krish's index is at:")
            print(f"         {FAISS_INDEX_PATH}")
        else:
            print(f"  [MODE] BM25 keyword filter ⚡  (forced)")
        top_candidates = pre_filter_candidates(candidates_path, all_skills, top_n=top_n)

    # ------------------------------------------------------------------
    # Step 3: Feature scoring
    # ------------------------------------------------------------------
    print(f"\n[STEP 3] Computing feature scores…")
    df = score_candidates(top_candidates, jd)

    # ------------------------------------------------------------------
    # Step 4: Save output
    # ------------------------------------------------------------------
    print(f"\n[STEP 4] Saving output…")
    save_parquet(df, output_path)

    wall_elapsed = time.time() - wall_start
    print(f"\n[DONE] Total wall-clock time: {wall_elapsed:.1f}s")
    print("=" * 60)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Muskan's Feature Engineering Pipeline — Redrob AI Hackathon"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES_PATH,
        help=f"Path to candidates.jsonl (default: {DEFAULT_CANDIDATES_PATH})",
    )
    parser.add_argument(
        "--jd",
        type=Path,
        default=DEFAULT_JD_PATH,
        help=f"Path to job_description.json (default: {DEFAULT_JD_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output Parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top candidates to keep after pre-filter (default: {DEFAULT_TOP_N})",
    )
    args = parser.parse_args()

    run_pipeline(
        candidates_path=args.candidates,
        jd_path=args.jd,
        output_path=args.output,
        top_n=args.top,
    )
