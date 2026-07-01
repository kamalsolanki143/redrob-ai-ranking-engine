# Validation Rules — Candidate Honeypot Detection System

## Overview

The validation system detects fake, inconsistent, and suspicious candidate profiles before they enter the final Top 100 ranking. It uses an **additive point-based risk scoring** system with multi-tier classification.

## Risk Tiers

| Tier | Risk Score | Action | Multiplier |
|------|-----------|--------|-----------|
| **honeypot** | ≥ 80 | Excluded from Top 100 | 0.0 |
| **suspicious** | 50–79 | Penalized score | 0.7 |
| **clean** | < 50 | No penalty | 1.0 |

## Architecture

```
src/validation/
├── __init__.py           ← Unified API: validate_candidate(), validate_candidates_batch()
├── _types.py             ← Shared dataclasses (ValidationFlag, ValidationResult)
├── profile_validator.py  ← Structural integrity checks
├── honeypot_detector.py  ← Impossible profile detection
└── anomaly_checker.py    ← Behavioral/statistical anomalies
```

---

## Module 1: Profile Validator (`profile_validator.py`)

Detects broken, invalid, or malformed profile data.

| Rule | Points | Description |
|------|--------|-------------|
| `missing_candidate_id` | +10 | candidate_id field is empty/missing |
| `missing_profile_headline` | +10 | Profile headline is empty |
| `missing_profile_years_of_experience` | +10 | years_of_experience is missing |
| `missing_profile_current_title` | +10 | current_title is missing |
| `empty_career_history` | +10 | No career history records at all |
| `invalid_start_date` | +10 | start_date not in YYYY-MM-DD format |
| `invalid_end_date` | +10 | end_date not in YYYY-MM-DD format |
| `negative_duration` | +15 | duration_months < 0 in a career role |
| `end_before_start` | +15 | end_date is before start_date in career |
| `negative_experience` | +20 | years_of_experience < 0 |
| `excessive_experience` | +20 | years_of_experience > 50 |
| `education_year_inverted` | +10 | Education end_year < start_year |
| `duplicate_career_entry` | +15 | Same company + title + start_date appears twice |
| `salary_range_inverted` | +5 | Salary min > max |

---

## Module 2: Honeypot Detector (`honeypot_detector.py`)

Detects impossible/fake profiles — the ~80 honeypots hidden in the dataset.

| Rule | Points | Description | Example |
|------|--------|-------------|---------|
| `expert_skill_zero_duration` | +30 each (cap 60) | Skill at "expert" with 0 months usage | "Python: expert, 0 months" |
| `mass_expert_near_zero` | +40 | 8+ expert skills with ≤3 months each | 10 expert skills, all 0-3 months |
| `experience_sum_exceeds_claimed` | +25 | Career months sum > claimed×1.5 | Claims 3 yrs but career sums to 8 yrs |
| `claimed_exceeds_career_sum` | +25 | Claimed experience > career sum×3 | Claims 10 yrs but only 2 yrs of roles |
| `future_start_date` | +20 | Career role starts after 2026-06-11 | start_date: "2027-01-01" |
| `underage_work_start` | +20 | Career starts before age 16 | Worked in 2005, education starts 2010 |
| `senior_title_trivial_experience` | +35 | Senior/Director/Architect with ≤1 yr | "Senior AI Architect" + 1 year exp |
| `duration_exceeds_timespan` | +20 | Role claims more months than date range allows | 36 months claimed, only 12 months between dates |
| `skill_count_explosion` | +25 | >20 skills at expert/advanced | 25 advanced/expert skills |
| `assessment_contradiction` | +20 | Expert skill but platform score <20 | "Python: expert" but scored 10/100 |
| `overlapping_careers` | +15 | Two roles overlap by >30 days | Two full-time jobs simultaneously |

**Reference date:** 2026-06-11 (dataset finalization date)

---

## Module 3: Anomaly Checker (`anomaly_checker.py`)

Detects behavioral and statistical anomalies suggesting fabrication.

| Rule | Points | Description | Example |
|------|--------|-------------|---------|
| `suspicious_career_jump` | +20 | Junior→Senior in one jump, <2 years | "Intern" → "VP Engineering" in 11 months |
| `unrealistic_growth` | +15 | <3 years exp but senior title | 2 years exp, title "Lead Architect" |
| `fake_activity_pattern` | +15 | >20 apps but 0% response + 0% interview | Bot-like mass application behavior |
| `endorsement_anomaly` | +10 | 0 months skill duration, >50 endorsements | "React: 0 months, 100 endorsements" |
| `completeness_content_mismatch` | +20 | >90% completeness but 0 skills, 0 career | Profile claims full but empty |
| `active_before_signup` | +15 | last_active_date < signup_date | Active in 2025 but signed up 2026 |
| `future_certification` | +10 | Certification year > 2026 | "AWS Cert 2030" |
| `premature_certification` | +10 | Certification before education start | Cert in 2010, education starts 2015 |
| `title_description_mismatch` | +10 | Technical title, non-technical description | Title: "ML Engineer", Description: "managed filing cabinets" |

---

## Usage

### Single Candidate Validation

```python
from src.validation import validate_candidate, get_penalty_multiplier

result = validate_candidate(candidate_dict)

print(result.candidate_id)    # "CAND_0042871"
print(result.is_honeypot)     # True/False
print(result.risk_score)      # 0-100
print(result.tier)            # "honeypot" / "suspicious" / "clean"
print(result.primary_reason)  # "10 expert skills with 0 months duration"
print(result.flags)           # List[ValidationFlag]

# Apply to scoring pipeline
multiplier = get_penalty_multiplier(result)
final_score *= multiplier
```

### Batch Validation

```python
from src.validation import validate_candidates_batch

df = validate_candidates_batch(candidates_list)
# Returns DataFrame with columns:
#   candidate_id, is_honeypot, risk_score, tier, primary_reason, flags
```

### Integration with Scoring Pipeline

The validation is integrated into `src/features/feature_combiner.py`. After computing
`skill_score`, `career_score`, and `signal_score`, the pipeline also runs validation
and adds `is_honeypot`, `risk_score`, and `validation_tier` columns to the output.

The penalty multiplier should be applied downstream in the ranker:
```python
# In scorer.py or ranker.py:
if row["is_honeypot"]:
    final_score = 0.0
elif row["validation_tier"] == "suspicious":
    final_score *= 0.7
```

---

## Output Format

For every candidate, the system returns:

```python
{
    "candidate_id": "CAND_0042871",
    "is_honeypot": True,
    "risk_score": 95,
    "tier": "honeypot",
    "primary_reason": "10 expert skill(s) with 0 months duration — impossible proficiency claim",
    "flags": ["expert_skill_zero_duration", "mass_expert_near_zero", "senior_title_trivial_experience"]
}
```

---

## Honeypot Examples Detected

### Example 1: Expert Skill Explosion
```
Profile: "Senior AI Architect", 1 year experience
Skills: 10 expert skills, all 0 months duration
Result: risk_score=100, tier=honeypot
Flags: expert_skill_zero_duration(60) + mass_expert_near_zero(40) + senior_title_trivial_experience(35)
```

### Example 2: Career Timeline Fraud
```
Profile: 3 years claimed experience
Career: 96 months total across 2 roles (8 years)
Result: risk_score=25+, tier depends on other flags
Flags: experience_sum_exceeds_claimed(25)
```

### Example 3: Impossible Progression
```
Profile: "VP Engineering", 2.5 years experience
Previous role: "Junior Intern" (11 months ago)
Result: risk_score=35, tier=clean (borderline, needs more flags)
Flags: suspicious_career_jump(20) + unrealistic_growth(15)
```

---

## Testing

```bash
pytest tests/test_honeypot.py -v
# 40 tests covering all validation rules, edge cases, and integration
```

---

## Design Decisions

1. **Additive scoring**: Each violation adds points independently. Multiple minor issues can compound into a honeypot classification.
2. **Capped at 100**: Even with many flags, risk_score never exceeds 100.
3. **Primary reason**: The highest-point flag becomes the primary_reason for quick understanding.
4. **Graceful degradation**: Missing/malformed fields never crash the validator — they may trigger validation flags but processing continues.
5. **Reference date**: Uses 2026-06-11 (dataset finalization date) for all "today" comparisons to ensure reproducibility.
6. **No external APIs**: Pure Python math, runs offline, no network calls.
