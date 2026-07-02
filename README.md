<p align="center">
  <h1 align="center">🤖 Redrob AI Ranking Engine</h1>
  <p align="center">Intelligent Candidate Discovery & Ranking System</p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square" />
    <img src="https://img.shields.io/badge/FAISS-CPU-green?style=flat-square" />
    <img src="https://img.shields.io/badge/sentence--transformers-MiniLM-orange?style=flat-square" />
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
    <img src="https://img.shields.io/badge/Hackathon-India%20Runs-red?style=flat-square" />
  </p>
</p>

---

> **Given 1,00,000 candidate profiles and one job description — find the best 100 candidates automatically. No paid APIs. No GPU. Runs in under 5 minutes on a normal laptop.**

Built for the **Redrob India Runs Hackathon — Track 1: Data & AI Challenge** | Prize Pool: ₹10 Lakhs

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Team](#-team)
- [Setup](#-setup)
- [How to Run](#-how-to-run)
- [Embeddings Pipeline](#-embeddings-pipeline-krrishs-part)
- [Scoring Formula](#-scoring-formula)
- [Output Format](#-output-format)
- [Constraints](#-constraints)
- [Tech Stack](#-tech-stack)

---

## 🎯 Problem Statement

Redrob is India's AI job platform with **1,00,000+ registered candidates**. When a company posts a job opening, recruiters manually sift through hundreds of profiles — missing great candidates due to:

- **Keyword filters** that don't understand meaning (missing "ML Engineer" ≈ "Machine Learning Developer")
- **No behavioral signals** — a profile that *looks* good but hasn't been active in 6 months is useless
- **No career intelligence** — no system reads career *trajectory*, only current title
- **Scale problem** — human review of 1 lakh profiles is impossible

**The question we answer:** Out of 1,00,000 candidates, who are the best 100 for this specific job?

---

## 💡 Our Solution

A **3-stage intelligent ranking pipeline** that combines semantic understanding, skill analysis, career trajectory scoring, and platform behavioral signals — all running on CPU in under 5 minutes.

```
Stage 1: Semantic Retrieval (FAISS)
  1,00,000 candidates → Top 1,000 by meaning-match

Stage 2: Multi-Signal Scoring
  Top 1,000 → scored on Skills + Career + Platform Signals

Stage 3: Final Ranking + Reasoning
  Top 1,000 → Top 100 with explainable reasoning per candidate
```

**What makes us different:**
- Semantic search understands *meaning*, not just keywords
- Career trajectory scoring (not just years of experience)
- Platform behavioral signals (are they actually active and responsive?)
- Honeypot detection (filters impossible/fake profiles automatically)
- Explainable reasoning per candidate (not a black box)

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                        OFFLINE PHASE (run once)                      ║
║                        No time limit                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   candidates.jsonl (487MB, ~1,00,000 profiles)                       ║
║          │                                                            ║
║          ▼                                                            ║
║   ┌─────────────────────┐                                            ║
║   │  build_text_blob.py │  Convert each candidate JSON               ║
║   │                     │  into one rich text string                  ║
║   └─────────┬───────────┘                                            ║
║             │                                                         ║
║             ▼                                                         ║
║   ┌──────────────────────────┐                                       ║
║   │ generate_embeddings.py   │  all-MiniLM-L6-v2                     ║
║   │                          │  Batch size: 512                       ║
║   │  100k candidates × 384d  │  Output: embeddings.npy               ║
║   └─────────┬────────────────┘                                       ║
║             │                                                         ║
║             ▼                                                         ║
║   ┌─────────────────────┐                                            ║
║   │   faiss_index.py    │  IndexFlatIP (cosine similarity)           ║
║   │                     │  Output: faiss_index.bin                   ║
║   └─────────────────────┘                                            ║
║                                                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                      ONLINE PHASE (< 5 minutes, CPU only)            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   job_description.json                                                ║
║          │                                                            ║
║          ▼                                                            ║
║   ┌─────────────────────┐                                            ║
║   │    parse_jd.py      │  Extract skills, requirements,             ║
║   │                     │  seniority, context from JD                ║
║   └─────────┬───────────┘                                            ║
║             │                                                         ║
║             ▼                                                         ║
║   ┌─────────────────────┐                                            ║
║   │   similarity.py     │  Embed JD → Search FAISS index             ║
║   │                     │  → Top 1,000 candidates                    ║
║   └─────────┬───────────┘         (semantic_score per candidate)     ║
║             │                                                         ║
║             ▼                                                         ║
║   ┌──────────────────────────────────────────────┐                  ║
║   │           MULTI-SIGNAL SCORER                │                  ║
║   │                                               │                  ║
║   │  skill_features.py    → skill_score (20%)    │                  ║
║   │  career_features.py   → career_score (15%)   │                  ║
║   │  signal_features.py   → signal_score (20%)   │                  ║
║   │  semantic (from FAISS) → sem_score   (45%)   │                  ║
║   │                                               │                  ║
║   │  honeypot_detector.py → filter fakes         │                  ║
║   └─────────┬─────────────────────────────────────┘                 ║
║             │                                                         ║
║             ▼                                                         ║
║   ┌──────────────────────────┐                                       ║
║   │  reasoning_generator.py  │  Per-candidate 1-2 sentence           ║
║   │                          │  explanation (no hallucination)        ║
║   └─────────┬────────────────┘                                       ║
║             │                                                         ║
║             ▼                                                         ║
║   ┌─────────────────────┐                                            ║
║   │   submission.csv    │  Top 100 candidates                        ║
║   │                     │  candidate_id, rank, score, reasoning      ║
║   └─────────────────────┘                                            ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
redrob-ai-ranking-engine/
│
├── data/
│   ├── raw/
│   │   ├── candidates.jsonl          ← main dataset (487MB, download separately)
│   │   ├── job_description.json      ← JD to rank candidates against
│   │   └── schema.json               ← candidate schema reference
│   │
│   ├── processed/
│   │   ├── candidate_embeddings.npy  ← generated by pipeline (offline)
│   │   ├── candidate_ids.pkl         ← generated by pipeline (offline)
│   │   └── processed_candidates.parquet
│   │
│   └── submissions/
│       └── ranked_candidates.csv     ← final output
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_profiles.py         ← cleans raw candidate JSON
│   │   ├── build_text_blob.py        ← ⭐ KRRISH: candidate → text string
│   │   └── parse_jd.py               ← parses job description
│   │
│   ├── embeddings/                   ← ⭐ KRRISH'S MODULE
│   │   ├── generate_embeddings.py    ← loads candidates, generates + saves embeddings
│   │   ├── faiss_index.py            ← builds FAISS index, saves to disk
│   │   └── similarity.py             ← embeds JD, searches index, returns top 1000
│   │
│   ├── features/
│   │   ├── semantic_features.py
│   │   ├── skill_features.py         ← skill match + proficiency + assessment scores
│   │   ├── career_features.py        ← trajectory, tenure, industry match
│   │   ├── signal_features.py        ← availability, responsiveness, github activity
│   │   └── feature_combiner.py       ← combines all scores into final score
│   │
│   ├── ranking/
│   │   ├── scorer.py                 ← applies weights, hard multipliers
│   │   ├── ranker.py                 ← sorts, produces top 100
│   │   └── reasoning_generator.py   ← generates per-candidate reasoning
│   │
│   ├── validation/
│   │   ├── honeypot_detector.py      ← detects impossible profiles
│   │   ├── profile_validator.py      ← validates candidate data
│   │   └── anomaly_checker.py        ← checks for data anomalies
│   │
│   └── utils/
│       ├── config.py                 ← all configuration values
│       ├── logger.py                 ← loguru logger setup
│       ├── helpers.py                ← utility functions
│       └── download_data.py          ← auto-downloads dataset if not present
│
├── models/
│   ├── faiss_index.bin               ← generated (offline)
│   ├── scaler.pkl                    ← generated (offline)
│   └── feature_metadata.json
│
├── outputs/
│   ├── top_100_candidates.csv        ← final submission file
│   ├── evaluation_report.json
│   └── feature_importance.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                  ← exploratory data analysis
│   ├── 02_embedding_generation.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_experiments.ipynb
│
├── tests/
│   ├── test_features.py
│   ├── test_ranker.py
│   └── test_honeypot.py
│
├── main.py                           ← single command to produce submission CSV
├── requirements.txt
├── submission_metadata.yaml
├── README.md
└── .gitignore
```

---

## 👥 Team

| Name | Role | Files |
|---|---|---|
| **Krrish Yaduka** | Embeddings + FAISS Retrieval | `build_text_blob.py`, `generate_embeddings.py`, `faiss_index.py`, `similarity.py` |
| **Kamal Solanki** | Feature Scoring + Ranking + Validation | `skill_features.py`, `career_features.py`, `signal_features.py`, `scorer.py`, `ranker.py`, `honeypot_detector.py` |

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/kamalsolanki143/redrob-ai-ranking-engine.git
cd redrob-ai-ranking-engine
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the dataset

`candidates.jsonl` is 487MB — too large for GitHub. Two options:

**Option A — Auto download (recommended)**
```bash
python main.py
# System detects missing file and downloads automatically
```

**Option B — Manual**
1. Download from: [Google Drive Link]
2. Place at: `data/raw/candidates.jsonl`

---

## 🚀 How to Run

### Single Command (Full Pipeline)

```bash
python rank.py --candidates ./data/raw/candidates.jsonl --out ./outputs/submission.csv
```

This produces the final `submission.csv` in under 5 minutes.

---

### Step by Step (If Needed)

**Step 1 — Generate Embeddings** *(run once, ~15 min for 100k candidates)*
```bash
python -m src.embeddings.generate_embeddings
```
Saves: `data/processed/candidate_embeddings.npy` + `candidate_ids.pkl`

**Step 2 — Build FAISS Index** *(run once, ~1 min)*
```bash
python -m src.embeddings.faiss_index
```
Saves: `models/faiss_index.bin`

**Step 3 — Run Ranking** *(under 5 minutes)*
```bash
python -m src.ranking.ranker
```
Saves: `outputs/top_100_candidates.csv`

**Step 4 — Validate Output**
```bash
python validate_submission.py outputs/top_100_candidates.csv
```

**Step 5 — Test on Sample First**
```bash
python main.py --sample
# Runs on sample_candidates.json (300KB) — fast testing
```

---

## 🧠 Embeddings Pipeline (Krrish's Part)

This is the core retrieval layer — converts 1,00,000 profiles into searchable vectors.

### File 1 — `build_text_blob.py`

Converts each candidate JSON into one text string for embedding.

**Input:**
```json
{
  "headline": "Senior ML Engineer",
  "skills": [{"name": "Python", "proficiency": "expert"}, ...],
  "career_history": [{"title": "ML Engineer", "description": "Built recommendation systems..."}]
}
```

**Output:**
```
"senior ml engineer machine learning python python python
 tensorflow pytorch data science built recommendation systems
 at scale deployed models to production b.tech computer science"
```

**Key rules:**
- Expert skills repeated **3×** (higher embedding weight)
- Advanced skills repeated **2×**
- Only last **3 career roles** included (recency matters)
- Name, location, company names **excluded** (removes bias)
- Text lowercased and cleaned
- Max 512 tokens (truncated if longer)

---

### File 2 — `generate_embeddings.py`

Generates and saves embeddings for all 1,00,000 candidates.

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

| Property | Value |
|---|---|
| Dimensions | 384 |
| Speed | ~3,000 candidates/sec on CPU |
| RAM for 100k | ~150MB |
| Total time | ~15 min (one time only) |

```bash
python -m src.embeddings.generate_embeddings
# Output: data/processed/candidate_embeddings.npy
#         data/processed/candidate_ids.pkl
```

---

### File 3 — `faiss_index.py`

Builds a FAISS index for fast similarity search.

**Index type:** `IndexFlatIP` (Inner Product = Cosine Similarity after L2 normalization)

| Property | Value |
|---|---|
| Index size | ~150MB for 100k vectors |
| Search time | ~0.1 sec for top 1000 |
| Exact search | Yes (no approximation) |

```bash
python -m src.embeddings.faiss_index
# Output: models/faiss_index.bin
```

---

### File 4 — `similarity.py`

Embeds the JD and retrieves top 1000 candidates.

```python
from src.embeddings.similarity import retrieve_top_candidates

top_1000_df = retrieve_top_candidates(top_k=1000)
# Returns: DataFrame with columns [candidate_id, semantic_score]
# This gets passed directly to the scoring pipeline
```

**Output shape:**
```
candidate_id   |  semantic_score
───────────────────────────────
CAND_0042871   |  0.987
CAND_0019884   |  0.973
...            |  ...
               |  (1000 rows total)
```

---

## 📊 Scoring Formula

```
Final Score =
  0.45 × semantic_score     ← JD embedding vs candidate embedding (FAISS)
+ 0.20 × skill_score        ← direct match + proficiency + assessment scores
+ 0.15 × career_score       ← progression, tenure, industry, education tier
+ 0.20 × signal_score       ← availability, responsiveness, github, completeness
```

**Hard multipliers (applied after scoring):**
```python
if not open_to_work_flag:           final_score *= 0.6
if days_since_active > 180:         final_score *= 0.7
if interview_completion_rate < 0.3: final_score *= 0.8
if is_honeypot:                     final_score  = 0.0
```

### Signal Features Used

| Signal | Weight | Meaning |
|---|---|---|
| `open_to_work_flag` | Hard filter | Not looking = penalized |
| `last_active_date` | High | Inactive 6mo+ = ghost |
| `recruiter_response_rate` | High | Will they reply? |
| `interview_completion_rate` | Medium | Do they show up? |
| `github_activity_score` | Medium | Real coder proof |
| `skill_assessment_scores` | High | Verified skill proof |
| `notice_period_days` | Low | How fast can they join? |
| `profile_completeness_score` | Low | Serious candidate? |

---

## 🍯 Honeypot Detection

The dataset contains **~80 fake profiles** with impossible data. If >10% of your top 100 are honeypots → **instant disqualification**.

Our system detects them automatically:

```python
# Flags that mark a honeypot:
✗ "Expert" skill with 0 months duration
✗ Total experience > company founding date gap
✗ 8+ skills all "expert" with near-zero usage
✗ Impossible date ranges in career history
```

Flagged candidates → forced below rank 100. Never in submission.

---

## 📋 Output Format

**File:** `outputs/top_100_candidates.csv`

```
candidate_id,    rank,  score,   reasoning
CAND_0042871,    1,     0.987,   "Senior ML Engineer, 7 yrs Python/TensorFlow, active on platform, open to work, Bangalore"
CAND_0019884,    2,     0.973,   "6 yrs applied ML, shipped vector search at scale, strong GitHub activity score 87/100"
...
CAND_0007729,    100,   0.412,   "Adjacent skills only, last active 4 months ago, partial JD match"
```

- Exactly **100 rows**
- Rank **1 = best**, 100 = last
- Score **monotonically non-increasing** (rank 1 has highest score)
- Reasoning references **specific facts** from profile (no hallucination)

---

## ⚡ Constraints Met

| Constraint | Limit | Our System |
|---|---|---|
| Runtime (ranking step) | ≤ 5 minutes | ~2-3 minutes ✅ |
| RAM | ≤ 16 GB | ~2 GB peak ✅ |
| Compute | CPU only | FAISS CPU index ✅ |
| Network | Offline | No API calls ✅ |
| Disk | ≤ 5 GB | ~700 MB ✅ |

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (faiss-cpu, IndexFlatIP) |
| Data Processing | pandas, numpy |
| Progress Bars | tqdm |
| Logging | loguru |
| Validation | pydantic |
| Demo UI | Streamlit on HuggingFace Spaces |

**Zero paid APIs. Fully offline after setup.**

---

## 📦 Requirements

```
sentence-transformers==2.7.0
faiss-cpu==1.8.0
pandas==2.2.0
numpy==1.26.0
scikit-learn==1.4.0
loguru==0.7.2
tqdm==4.66.0
pydantic==2.6.0
gdown==5.1.0
fastapi==0.110.0
uvicorn==0.27.0
streamlit==1.32.0
```

```bash
pip install -r requirements.txt
```

---

## 🗂️ Dataset

| File | Size | Description |
|---|---|---|
| `candidates.jsonl` | 487 MB | ~1,00,000 candidate profiles |
| `candidate_schema.json` | 9 KB | Field definitions |
| `job_description.json` | 40 KB | The JD to rank against |
| `sample_candidates.json` | 300 KB | Small sample for testing |

Dataset provided by Redrob AI — India Runs Hackathon bundle.
Download: [Google Drive Link — paste here]

---

## ✅ Submission Checklist

- [ ] `outputs/top_100_candidates.csv` — exactly 100 rows, ranks 1-100
- [ ] Scores monotonically non-increasing
- [ ] No duplicate `candidate_id`
- [ ] All `candidate_id` values exist in `candidates.jsonl`
- [ ] `python validate_submission.py` passes with 0 errors
- [ ] GitHub repo public + README complete
- [ ] `submission_metadata.yaml` filled
- [ ] HuggingFace / Streamlit demo live and working
- [ ] Single command in README produces submission CSV

---

## 🏆 Hackathon

**India Runs Hackathon** by Redrob AI × Hack2Skill
Track 1: Intelligent Candidate Discovery & Ranking
Prize Pool: ₹10 Lakhs
Submission Deadline: 2 july 2026

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
