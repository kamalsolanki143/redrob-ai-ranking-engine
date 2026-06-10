from __future__ import annotations

import re
import json
from pathlib import Path

SKILL_REPEAT = {
    "expert": 3,
    "advanced": 2,
    "intermediate": 1,
    "beginner": 1,
}

MAX_CHARS = 2048


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def build_candidate_text(candidate: dict) -> str:
    parts = []

    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    current_title = profile.get("current_title", "")
    current_industry = profile.get("current_industry", "")

    if headline:
        parts.append(_clean(headline))
    if summary:
        parts.append(_clean(summary))
    if current_title:
        parts.append(_clean(current_title))
    if current_industry:
        parts.append(_clean(current_industry))

    skills = candidate.get("skills", [])
    skill_parts = []
    for skill in skills:
        name = skill.get("name", "")
        if not name:
            continue
        name_clean = _clean(name)
        prof = skill.get("proficiency", "intermediate")
        repeat = SKILL_REPEAT.get(prof, 1)
        for _ in range(repeat):
            skill_parts.append(name_clean)
    if skill_parts:
        parts.append(" ".join(skill_parts))

    career = candidate.get("career_history", [])
    for role in career[-3:]:
        title = role.get("title", "")
        desc = role.get("description", "")
        role_text = f"{title} {desc}".strip()
        if role_text:
            parts.append(_clean(role_text))

    education = candidate.get("education", [])
    edu_parts = []
    for edu in education:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        if degree:
            edu_parts.append(_clean(degree))
        if field:
            edu_parts.append(_clean(field))
    if edu_parts:
        parts.append(" ".join(edu_parts))

    certs = candidate.get("certifications", [])
    cert_parts = []
    for cert in certs:
        name = cert.get("name", "")
        if name:
            cert_parts.append(_clean(name))
    if cert_parts:
        parts.append(" ".join(cert_parts))

    text = " ".join(part for part in parts if part)
    return _truncate(text)


def build_jd_text(jd: dict) -> str:
    parts = []

    title = jd.get("title", "")
    description = jd.get("description", "")
    responsibilities = jd.get("responsibilities", "")
    if title:
        parts.append(_clean(title))
    if description:
        parts.append(_clean(description))
    if responsibilities:
        parts.append(_clean(responsibilities))

    required_skills = jd.get("required_skills", [])
    preferred_skills = jd.get("preferred_skills", [])
    skill_parts = []
    for skill_list in [required_skills, preferred_skills]:
        if not isinstance(skill_list, list):
            continue
        for skill in skill_list:
            if isinstance(skill, dict):
                name = skill.get("name", "")
                importance = skill.get("importance", "required")
            else:
                name = str(skill)
                importance = "required"
            if not name:
                continue
            name_clean = _clean(name)
            repeat = 3 if importance == "required" else 2
            for _ in range(repeat):
                skill_parts.append(name_clean)
    if skill_parts:
        parts.append(" ".join(skill_parts))

    qualifications = jd.get("qualifications", [])
    if isinstance(qualifications, list):
        for q in qualifications:
            if isinstance(q, dict):
                q_text = f"{q.get('degree', '')} {q.get('field', '')}"
            else:
                q_text = str(q)
            if q_text.strip():
                parts.append(_clean(q_text))

    text = " ".join(part for part in parts if part)
    return _truncate(text)


if __name__ == "__main__":
    sample = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "headline": "Backend Engineer | SQL, Spark, Cloud",
            "summary": "Data professional with 6.9 years of experience.",
            "current_title": "Backend Engineer",
            "current_industry": "IT Services",
        },
        "skills": [
            {"name": "Python", "proficiency": "expert"},
            {"name": "SQL", "proficiency": "advanced"},
            {"name": "Java", "proficiency": "beginner"},
        ],
        "career_history": [
            {"title": "Backend Engineer", "description": "Built data pipelines with Spark."},
            {"title": "Analytics Engineer", "description": "Built Airflow pipelines."},
        ],
        "education": [{"degree": "B.E.", "field_of_study": "Computer Science"}],
        "certifications": [{"name": "AWS Certified"}],
    }
    result = build_candidate_text(sample)
    print("=== build_candidate_text output ===")
    print(result)
    print()

    jd_sample = {
        "title": "Senior ML Engineer",
        "description": "Build and deploy ML models at scale.",
        "required_skills": [
            {"name": "Python", "importance": "required"},
            {"name": "PyTorch", "importance": "required"},
            {"name": "Kubernetes", "importance": "preferred"},
        ],
        "qualifications": [{"degree": "B.Tech", "field": "Computer Science"}],
    }
    jd_result = build_jd_text(jd_sample)
    print("=== build_jd_text output ===")
    print(jd_result)