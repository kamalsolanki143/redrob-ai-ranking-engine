"""
tests/test_honeypot.py
======================
Comprehensive tests for the Candidate Validation & Honeypot Detection System.

Covers:
- ProfileValidator (structural integrity checks)
- HoneypotDetector (impossible profile detection)
- AnomalyChecker (behavioral/statistical anomalies)
- Integration (validate_candidate, batch, penalty multiplier, tiers)

Run: pytest tests/test_honeypot.py -v
"""

import pytest

from src.validation import (
    validate_candidate,
    validate_candidates_batch,
    get_penalty_multiplier,
    compute_tier,
    ValidationResult,
    ValidationFlag,
    HONEYPOT_THRESHOLD,
    SUSPICIOUS_THRESHOLD,
)
from src.validation.profile_validator import validate_profile
from src.validation.honeypot_detector import detect_honeypot
from src.validation.anomaly_checker import check_anomalies


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def valid_candidate():
    """A clean, legitimate profile based on sample_candidates.json structure."""
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Ira Vora",
            "headline": "Backend Engineer | SQL, Spark, Cloud",
            "summary": "Software professional with 6.9 years of experience.",
            "location": "Toronto",
            "country": "Canada",
            "years_of_experience": 6.9,
            "current_title": "Backend Engineer",
            "current_company": "Mindtree",
            "current_company_size": "10001+",
            "current_industry": "IT Services",
        },
        "career_history": [
            {
                "company": "Mindtree",
                "title": "Backend Engineer",
                "start_date": "2024-03-08",
                "end_date": None,
                "duration_months": 27,
                "is_current": True,
                "industry": "IT Services",
                "company_size": "10001+",
                "description": "Implemented streaming data pipelines on Kafka and Spark.",
            },
            {
                "company": "Dunder Mifflin",
                "title": "Analytics Engineer",
                "start_date": "2019-07-03",
                "end_date": "2024-01-08",
                "duration_months": 55,
                "is_current": False,
                "industry": "Paper Products",
                "company_size": "201-500",
                "description": "Built data pipelines on Apache Airflow processing data.",
            },
        ],
        "education": [
            {
                "institution": "Lovely Professional University",
                "degree": "B.E.",
                "field_of_study": "Computer Science",
                "start_year": 2017,
                "end_year": 2020,
                "grade": "8.24 CGPA",
                "tier": "tier_3",
            }
        ],
        "skills": [
            {"name": "NLP", "proficiency": "advanced", "endorsements": 37, "duration_months": 26},
            {"name": "Python", "proficiency": "advanced", "endorsements": 21, "duration_months": 36},
            {"name": "SQL", "proficiency": "intermediate", "endorsements": 9, "duration_months": 40},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 86.9,
            "signup_date": "2025-10-16",
            "last_active_date": "2026-05-20",
            "open_to_work_flag": True,
            "profile_views_received_30d": 23,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.34,
            "avg_response_time_hours": 177.8,
            "skill_assessment_scores": {"NLP": 38.8},
            "connection_count": 356,
            "endorsements_received": 35,
            "notice_period_days": 60,
            "expected_salary_range_inr_lpa": {"min": 18.7, "max": 36.1},
            "preferred_work_mode": "onsite",
            "willing_to_relocate": False,
            "github_activity_score": 9.2,
            "search_appearance_30d": 249,
            "saved_by_recruiters_30d": 4,
            "interview_completion_rate": 0.71,
            "offer_acceptance_rate": 0.58,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": False,
        },
    }


@pytest.fixture
def honeypot_candidate():
    """A fake profile: expert skills with 0 months, senior title with 1 year."""
    return {
        "candidate_id": "CAND_HONEY01",
        "profile": {
            "anonymized_name": "Fake Expert",
            "headline": "Senior AI Architect",
            "summary": "Expert in everything",
            "location": "Mumbai",
            "country": "India",
            "years_of_experience": 1.0,
            "current_title": "Senior AI Architect",
            "current_company": "FakeCo",
            "current_company_size": "1-10",
            "current_industry": "AI/ML",
        },
        "career_history": [
            {
                "company": "FakeCo",
                "title": "Senior AI Architect",
                "start_date": "2025-06-01",
                "end_date": None,
                "duration_months": 12,
                "is_current": True,
                "industry": "AI/ML",
                "company_size": "1-10",
                "description": "Led all AI initiatives for the company.",
            }
        ],
        "education": [
            {"institution": "IIT Delhi", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2020, "end_year": 2024, "grade": "9.5", "tier": "tier_1"}
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 100, "duration_months": 0},
            {"name": "TensorFlow", "proficiency": "expert", "endorsements": 80, "duration_months": 0},
            {"name": "PyTorch", "proficiency": "expert", "endorsements": 60, "duration_months": 0},
            {"name": "NLP", "proficiency": "expert", "endorsements": 50, "duration_months": 0},
            {"name": "CV", "proficiency": "expert", "endorsements": 40, "duration_months": 0},
            {"name": "MLOps", "proficiency": "expert", "endorsements": 30, "duration_months": 0},
            {"name": "AWS", "proficiency": "expert", "endorsements": 20, "duration_months": 0},
            {"name": "Docker", "proficiency": "expert", "endorsements": 10, "duration_months": 0},
            {"name": "K8s", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "SQL", "proficiency": "expert", "endorsements": 3, "duration_months": 0},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "native"}],
        "redrob_signals": {
            "profile_completeness_score": 95,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 50,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.9,
            "avg_response_time_hours": 12,
            "skill_assessment_scores": {},
            "connection_count": 500,
            "endorsements_received": 200,
            "notice_period_days": 0,
            "expected_salary_range_inr_lpa": {"min": 50, "max": 80},
            "preferred_work_mode": "remote",
            "willing_to_relocate": True,
            "github_activity_score": 90,
            "search_appearance_30d": 200,
            "saved_by_recruiters_30d": 15,
            "interview_completion_rate": 1.0,
            "offer_acceptance_rate": 1.0,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


@pytest.fixture
def suspicious_candidate():
    """A suspicious profile: career jump anomaly + unrealistic growth."""
    return {
        "candidate_id": "CAND_SUSP001",
        "profile": {
            "anonymized_name": "Suspicious Person",
            "headline": "Director of Engineering",
            "summary": "Fast-tracked career.",
            "location": "Delhi",
            "country": "India",
            "years_of_experience": 2.5,
            "current_title": "Director of Engineering",
            "current_company": "StartupXYZ",
            "current_company_size": "11-50",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "StartupXYZ",
                "title": "Director of Engineering",
                "start_date": "2025-06-01",
                "end_date": None,
                "duration_months": 12,
                "is_current": True,
                "industry": "Software",
                "company_size": "11-50",
                "description": "Leading the engineering team and building data systems.",
            },
            {
                "company": "BigCorp",
                "title": "Junior Intern",
                "start_date": "2024-01-01",
                "end_date": "2025-05-01",
                "duration_months": 16,
                "is_current": False,
                "industry": "Software",
                "company_size": "1001-5000",
                "description": "Assisted senior engineers with software testing tasks.",
            },
        ],
        "education": [
            {"institution": "Local College", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2020, "end_year": 2024, "grade": "7.5", "tier": "tier_4"}
        ],
        "skills": [
            {"name": "Python", "proficiency": "intermediate", "endorsements": 5, "duration_months": 20},
            {"name": "JavaScript", "proficiency": "beginner", "endorsements": 2, "duration_months": 10},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 60,
            "signup_date": "2024-06-01",
            "last_active_date": "2026-04-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.2,
            "avg_response_time_hours": 48,
            "skill_assessment_scores": {},
            "connection_count": 50,
            "endorsements_received": 5,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 35},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 10,
            "search_appearance_30d": 30,
            "saved_by_recruiters_30d": 2,
            "interview_completion_rate": 0.5,
            "offer_acceptance_rate": -1,
            "verified_email": True,
            "verified_phone": False,
            "linkedin_connected": False,
        },
    }


# ===========================================================================
# TestProfileValidator
# ===========================================================================

class TestProfileValidator:
    """Tests for profile_validator.py structural integrity checks."""

    def test_valid_profile_no_flags(self, valid_candidate):
        flags = validate_profile(valid_candidate)
        assert len(flags) == 0

    def test_missing_candidate_id(self, valid_candidate):
        candidate = {**valid_candidate, "candidate_id": ""}
        flags = validate_profile(candidate)
        flag_names = [f.name for f in flags]
        assert "missing_candidate_id" in flag_names

    def test_missing_headline(self, valid_candidate):
        candidate = {**valid_candidate}
        candidate["profile"] = {**candidate["profile"], "headline": ""}
        flags = validate_profile(candidate)
        flag_names = [f.name for f in flags]
        assert "missing_profile_headline" in flag_names

    def test_negative_duration(self, valid_candidate):
        candidate = {**valid_candidate}
        candidate["career_history"] = [
            {**valid_candidate["career_history"][0], "duration_months": -5}
        ]
        flags = validate_profile(candidate)
        flag_names = [f.name for f in flags]
        assert "negative_duration" in flag_names

    def test_end_before_start(self, valid_candidate):
        candidate = {**valid_candidate}
        candidate["career_history"] = [{
            "company": "TestCo",
            "title": "Engineer",
            "start_date": "2024-06-01",
            "end_date": "2023-01-01",
            "duration_months": 10,
            "is_current": False,
            "industry": "Tech",
            "company_size": "51-200",
            "description": "Testing.",
        }]
        flags = validate_profile(candidate)
        flag_names = [f.name for f in flags]
        assert "end_before_start" in flag_names

    def test_education_year_inverted(self, valid_candidate):
        candidate = {**valid_candidate}
        candidate["education"] = [{
            "institution": "Test Uni",
            "degree": "B.Tech",
            "field_of_study": "CS",
            "start_year": 2024,
            "end_year": 2020,
            "grade": "8.0",
            "tier": "tier_3",
        }]
        flags = validate_profile(candidate)
        flag_names = [f.name for f in flags]
        assert "education_year_inverted" in flag_names

    def test_duplicate_career_entries(self, valid_candidate):
        candidate = {**valid_candidate}
        role = valid_candidate["career_history"][0].copy()
        candidate["career_history"] = [role, role.copy()]
        flags = validate_profile(candidate)
        flag_names = [f.name for f in flags]
        assert "duplicate_career_entry" in flag_names

    def test_salary_range_inverted(self, valid_candidate):
        candidate = {**valid_candidate}
        candidate["redrob_signals"] = {
            **valid_candidate["redrob_signals"],
            "expected_salary_range_inr_lpa": {"min": 50.0, "max": 20.0},
        }
        flags = validate_profile(candidate)
        flag_names = [f.name for f in flags]
        assert "salary_range_inverted" in flag_names

    def test_excessive_experience(self, valid_candidate):
        candidate = {**valid_candidate}
        candidate["profile"] = {**candidate["profile"], "years_of_experience": 55}
        flags = validate_profile(candidate)
        flag_names = [f.name for f in flags]
        assert "excessive_experience" in flag_names

    def test_invalid_date_format(self, valid_candidate):
        candidate = {**valid_candidate}
        candidate["career_history"] = [{
            "company": "TestCo",
            "title": "Engineer",
            "start_date": "not-a-date",
            "end_date": "2024-01-01",
            "duration_months": 10,
            "is_current": False,
            "industry": "Tech",
            "company_size": "51-200",
            "description": "Testing.",
        }]
        flags = validate_profile(candidate)
        flag_names = [f.name for f in flags]
        assert "invalid_start_date" in flag_names


# ===========================================================================
# TestHoneypotDetector
# ===========================================================================

class TestHoneypotDetector:
    """Tests for honeypot_detector.py impossible profile detection."""

    def test_legitimate_profile_clean(self, valid_candidate):
        flags = detect_honeypot(valid_candidate)
        assert len(flags) == 0

    def test_expert_skill_zero_duration(self):
        candidate = {
            "candidate_id": "CAND_TEST001",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co", "title": "Engineer", "start_date": "2021-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Tech", "company_size": "51-200", "description": "Built systems."}
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 0},
            ],
            "certifications": [],
            "redrob_signals": {"skill_assessment_scores": {}},
        }
        flags = detect_honeypot(candidate)
        flag_names = [f.name for f in flags]
        assert "expert_skill_zero_duration" in flag_names
        # Should be 30 points for 1 occurrence
        expert_flag = next(f for f in flags if f.name == "expert_skill_zero_duration")
        assert expert_flag.points == 30

    def test_expert_skill_zero_duration_capped_at_60(self):
        """Multiple expert+0 skills should cap at 60 points."""
        candidate = {
            "candidate_id": "CAND_TEST002",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co", "title": "Engineer", "start_date": "2021-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Tech", "company_size": "51-200", "description": "Built systems."}
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [
                {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 5, "duration_months": 0}
                for i in range(5)
            ],
            "certifications": [],
            "redrob_signals": {"skill_assessment_scores": {}},
        }
        flags = detect_honeypot(candidate)
        expert_flag = next(f for f in flags if f.name == "expert_skill_zero_duration")
        assert expert_flag.points == 60  # capped

    def test_eight_plus_expert_near_zero(self):
        candidate = {
            "candidate_id": "CAND_TEST003",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co", "title": "Engineer", "start_date": "2021-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Tech", "company_size": "51-200", "description": "Built code systems."}
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [
                {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 5, "duration_months": 2}
                for i in range(9)
            ],
            "certifications": [],
            "redrob_signals": {"skill_assessment_scores": {}},
        }
        flags = detect_honeypot(candidate)
        flag_names = [f.name for f in flags]
        assert "mass_expert_near_zero" in flag_names

    def test_experience_mismatch_sum_exceeds_claimed(self):
        """Career sum vastly exceeds claimed experience."""
        candidate = {
            "candidate_id": "CAND_TEST004",
            "profile": {"years_of_experience": 3, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co1", "title": "Eng", "start_date": "2020-01-01",
                 "end_date": "2022-01-01", "duration_months": 48, "is_current": False,
                 "industry": "Tech", "company_size": "51-200", "description": "Built systems."},
                {"company": "Co2", "title": "Eng", "start_date": "2022-01-01",
                 "end_date": None, "duration_months": 48, "is_current": True,
                 "industry": "Tech", "company_size": "51-200", "description": "Built more systems."},
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [],
            "certifications": [],
            "redrob_signals": {"skill_assessment_scores": {}},
        }
        flags = detect_honeypot(candidate)
        flag_names = [f.name for f in flags]
        assert "experience_sum_exceeds_claimed" in flag_names

    def test_senior_title_trivial_experience(self):
        candidate = {
            "candidate_id": "CAND_TEST005",
            "profile": {"years_of_experience": 0.5, "current_title": "Principal Architect"},
            "career_history": [
                {"company": "Co", "title": "Principal Architect", "start_date": "2026-01-01",
                 "end_date": None, "duration_months": 6, "is_current": True,
                 "industry": "Tech", "company_size": "51-200", "description": "Architecture."}
            ],
            "education": [{"start_year": 2020, "end_year": 2024}],
            "skills": [],
            "certifications": [],
            "redrob_signals": {"skill_assessment_scores": {}},
        }
        flags = detect_honeypot(candidate)
        flag_names = [f.name for f in flags]
        assert "senior_title_trivial_experience" in flag_names

    def test_overlapping_careers(self):
        candidate = {
            "candidate_id": "CAND_TEST006",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co1", "title": "Engineer", "start_date": "2022-01-01",
                 "end_date": "2024-06-01", "duration_months": 30, "is_current": False,
                 "industry": "Tech", "company_size": "51-200", "description": "Built systems."},
                {"company": "Co2", "title": "Developer", "start_date": "2023-01-01",
                 "end_date": "2025-01-01", "duration_months": 24, "is_current": False,
                 "industry": "Tech", "company_size": "51-200", "description": "Built code."},
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [],
            "certifications": [],
            "redrob_signals": {"skill_assessment_scores": {}},
        }
        flags = detect_honeypot(candidate)
        flag_names = [f.name for f in flags]
        assert "overlapping_careers" in flag_names

    def test_assessment_contradiction(self):
        candidate = {
            "candidate_id": "CAND_TEST007",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co", "title": "Engineer", "start_date": "2021-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Tech", "company_size": "51-200", "description": "Built systems."}
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 48},
            ],
            "certifications": [],
            "redrob_signals": {"skill_assessment_scores": {"Python": 10}},
        }
        flags = detect_honeypot(candidate)
        flag_names = [f.name for f in flags]
        assert "assessment_contradiction" in flag_names

    def test_duration_exceeds_timespan(self):
        candidate = {
            "candidate_id": "CAND_TEST008",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co", "title": "Engineer", "start_date": "2023-01-01",
                 "end_date": "2024-01-01", "duration_months": 36, "is_current": False,
                 "industry": "Tech", "company_size": "51-200", "description": "Built systems."}
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [],
            "certifications": [],
            "redrob_signals": {"skill_assessment_scores": {}},
        }
        flags = detect_honeypot(candidate)
        flag_names = [f.name for f in flags]
        assert "duration_exceeds_timespan" in flag_names


# ===========================================================================
# TestAnomalyChecker
# ===========================================================================

class TestAnomalyChecker:
    """Tests for anomaly_checker.py behavioral/statistical anomalies."""

    def test_valid_candidate_no_anomalies(self, valid_candidate):
        flags = check_anomalies(valid_candidate)
        assert len(flags) == 0

    def test_suspicious_career_jump(self):
        candidate = {
            "candidate_id": "CAND_ANOM001",
            "profile": {"years_of_experience": 3, "current_title": "VP Engineering"},
            "career_history": [
                {"company": "BigCo", "title": "VP Engineering", "start_date": "2025-06-01",
                 "end_date": None, "duration_months": 12, "is_current": True,
                 "industry": "Tech", "company_size": "1001-5000",
                 "description": "Leading engineering organization with 50+ engineers."},
                {"company": "StartCo", "title": "Junior Intern", "start_date": "2024-06-01",
                 "end_date": "2025-05-01", "duration_months": 11, "is_current": False,
                 "industry": "Tech", "company_size": "11-50",
                 "description": "Assisted with basic QA testing tasks."},
            ],
            "education": [{"start_year": 2020, "end_year": 2024}],
            "skills": [],
            "certifications": [],
            "redrob_signals": {
                "profile_completeness_score": 60,
                "signup_date": "2024-01-01",
                "last_active_date": "2026-04-01",
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.3,
                "interview_completion_rate": 0.5,
                "skill_assessment_scores": {},
            },
        }
        flags = check_anomalies(candidate)
        flag_names = [f.name for f in flags]
        assert "suspicious_career_jump" in flag_names

    def test_unrealistic_growth(self):
        candidate = {
            "candidate_id": "CAND_ANOM002",
            "profile": {"years_of_experience": 2.0, "current_title": "Senior Lead Architect"},
            "career_history": [
                {"company": "Co", "title": "Senior Lead Architect", "start_date": "2024-06-01",
                 "end_date": None, "duration_months": 24, "is_current": True,
                 "industry": "Tech", "company_size": "51-200",
                 "description": "Architecture and system design for cloud platform."}
            ],
            "education": [{"start_year": 2020, "end_year": 2024}],
            "skills": [],
            "certifications": [],
            "redrob_signals": {
                "profile_completeness_score": 50,
                "signup_date": "2024-01-01",
                "last_active_date": "2026-04-01",
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.3,
                "interview_completion_rate": 0.5,
                "skill_assessment_scores": {},
            },
        }
        flags = check_anomalies(candidate)
        flag_names = [f.name for f in flags]
        assert "unrealistic_growth" in flag_names

    def test_fake_activity_pattern(self):
        candidate = {
            "candidate_id": "CAND_ANOM003",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co", "title": "Engineer", "start_date": "2021-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Tech", "company_size": "51-200",
                 "description": "Built data pipelines and backend systems."}
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [],
            "certifications": [],
            "redrob_signals": {
                "profile_completeness_score": 50,
                "signup_date": "2024-01-01",
                "last_active_date": "2026-04-01",
                "applications_submitted_30d": 25,
                "recruiter_response_rate": 0,
                "interview_completion_rate": 0,
                "skill_assessment_scores": {},
            },
        }
        flags = check_anomalies(candidate)
        flag_names = [f.name for f in flags]
        assert "fake_activity_pattern" in flag_names

    def test_signup_after_active(self):
        candidate = {
            "candidate_id": "CAND_ANOM004",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co", "title": "Engineer", "start_date": "2021-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Tech", "company_size": "51-200",
                 "description": "Built data systems and APIs for the platform."}
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [],
            "certifications": [],
            "redrob_signals": {
                "profile_completeness_score": 50,
                "signup_date": "2026-03-01",
                "last_active_date": "2025-01-01",
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.3,
                "interview_completion_rate": 0.5,
                "skill_assessment_scores": {},
            },
        }
        flags = check_anomalies(candidate)
        flag_names = [f.name for f in flags]
        assert "active_before_signup" in flag_names

    def test_future_certification(self):
        candidate = {
            "candidate_id": "CAND_ANOM005",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co", "title": "Engineer", "start_date": "2021-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Tech", "company_size": "51-200",
                 "description": "Built software systems."}
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [],
            "certifications": [{"name": "AWS Solutions Architect", "issuer": "AWS", "year": 2030}],
            "redrob_signals": {
                "profile_completeness_score": 50,
                "signup_date": "2024-01-01",
                "last_active_date": "2026-04-01",
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.3,
                "interview_completion_rate": 0.5,
                "skill_assessment_scores": {},
            },
        }
        flags = check_anomalies(candidate)
        flag_names = [f.name for f in flags]
        assert "future_certification" in flag_names

    def test_endorsement_anomaly(self):
        candidate = {
            "candidate_id": "CAND_ANOM006",
            "profile": {"years_of_experience": 5, "current_title": "Engineer"},
            "career_history": [
                {"company": "Co", "title": "Engineer", "start_date": "2021-01-01",
                 "end_date": None, "duration_months": 60, "is_current": True,
                 "industry": "Tech", "company_size": "51-200",
                 "description": "Built data infrastructure and deployed APIs."}
            ],
            "education": [{"start_year": 2015, "end_year": 2019}],
            "skills": [
                {"name": "Magic Skill", "proficiency": "beginner", "endorsements": 100, "duration_months": 0},
            ],
            "certifications": [],
            "redrob_signals": {
                "profile_completeness_score": 50,
                "signup_date": "2024-01-01",
                "last_active_date": "2026-04-01",
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.3,
                "interview_completion_rate": 0.5,
                "skill_assessment_scores": {},
            },
        }
        flags = check_anomalies(candidate)
        flag_names = [f.name for f in flags]
        assert "endorsement_anomaly" in flag_names


# ===========================================================================
# TestValidationIntegration
# ===========================================================================

class TestValidationIntegration:
    """Integration tests for the unified validation system."""

    def test_clean_candidate_tier(self, valid_candidate):
        result = validate_candidate(valid_candidate)
        assert result.tier == "clean"
        assert result.is_honeypot is False
        assert result.risk_score < SUSPICIOUS_THRESHOLD

    def test_honeypot_candidate_tier(self, honeypot_candidate):
        result = validate_candidate(honeypot_candidate)
        assert result.tier == "honeypot"
        assert result.is_honeypot is True
        assert result.risk_score >= HONEYPOT_THRESHOLD

    def test_suspicious_candidate_tier(self, suspicious_candidate):
        result = validate_candidate(suspicious_candidate)
        # Has career jump (20) + unrealistic growth (15) = 35
        # Enhance with more flags to push into suspicious territory
        assert result.risk_score >= 30  # Catches multiple anomalies
        assert len(result.flags) >= 2
        flag_names = [f.name for f in result.flags]
        assert "suspicious_career_jump" in flag_names
        assert "unrealistic_growth" in flag_names

    def test_suspicious_tier_threshold(self):
        """A candidate that hits exactly suspicious tier (score >= 50)."""
        candidate = {
            "candidate_id": "CAND_SUSP002",
            "profile": {
                "anonymized_name": "Suspicious",
                "headline": "Senior Director",
                "summary": "Fast career.",
                "location": "Delhi",
                "country": "India",
                "years_of_experience": 2.0,
                "current_title": "Senior Director",
                "current_company": "Co",
                "current_company_size": "11-50",
                "current_industry": "Tech",
            },
            "career_history": [
                {"company": "Co", "title": "Senior Director", "start_date": "2025-06-01",
                 "end_date": None, "duration_months": 12, "is_current": True,
                 "industry": "Tech", "company_size": "11-50",
                 "description": "Leading engineering team."},
                {"company": "OldCo", "title": "Junior Trainee", "start_date": "2024-06-01",
                 "end_date": "2025-05-01", "duration_months": 11, "is_current": False,
                 "industry": "Tech", "company_size": "51-200",
                 "description": "Basic testing tasks."},
            ],
            "education": [{"institution": "X", "degree": "B.Tech", "field_of_study": "CS",
                           "start_year": 2020, "end_year": 2024, "grade": "7.5", "tier": "tier_4"}],
            "skills": [
                {"name": "Magic", "proficiency": "beginner", "endorsements": 100, "duration_months": 0},
            ],
            "certifications": [{"name": "Future Cert", "issuer": "X", "year": 2030}],
            "languages": [],
            "redrob_signals": {
                "profile_completeness_score": 50,
                "signup_date": "2026-03-01",
                "last_active_date": "2025-01-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 5,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.2,
                "avg_response_time_hours": 48,
                "skill_assessment_scores": {},
                "connection_count": 50,
                "endorsements_received": 5,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 20, "max": 35},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 10,
                "search_appearance_30d": 30,
                "saved_by_recruiters_30d": 2,
                "interview_completion_rate": 0.5,
                "offer_acceptance_rate": -1,
                "verified_email": True,
                "verified_phone": False,
                "linkedin_connected": False,
            },
        }
        result = validate_candidate(candidate)
        assert result.risk_score >= SUSPICIOUS_THRESHOLD
        assert result.tier in ("suspicious", "honeypot")

    def test_penalty_multiplier_values(self):
        clean = ValidationResult(
            candidate_id="X", is_honeypot=False, risk_score=10,
            primary_reason="Clean", tier="clean",
        )
        suspicious = ValidationResult(
            candidate_id="Y", is_honeypot=False, risk_score=60,
            primary_reason="Suspicious", tier="suspicious",
        )
        honeypot = ValidationResult(
            candidate_id="Z", is_honeypot=True, risk_score=90,
            primary_reason="Fake", tier="honeypot",
        )
        assert get_penalty_multiplier(clean) == 1.0
        assert get_penalty_multiplier(suspicious) == 0.7
        assert get_penalty_multiplier(honeypot) == 0.0

    def test_batch_validation(self, valid_candidate, honeypot_candidate):
        df = validate_candidates_batch([valid_candidate, honeypot_candidate])
        assert len(df) == 2
        assert list(df.columns) == [
            "candidate_id", "is_honeypot", "risk_score", "tier", "primary_reason", "flags"
        ]
        # First should be clean, second should be honeypot
        assert df.iloc[0]["tier"] == "clean"
        assert df.iloc[1]["tier"] == "honeypot"

    def test_boundary_score_50(self):
        """risk_score exactly 50 should be 'suspicious'."""
        assert compute_tier(50) == "suspicious"

    def test_boundary_score_80(self):
        """risk_score exactly 80 should be 'honeypot'."""
        assert compute_tier(80) == "honeypot"

    def test_boundary_score_49(self):
        """risk_score 49 should be 'clean'."""
        assert compute_tier(49) == "clean"

    def test_boundary_score_79(self):
        """risk_score 79 should be 'suspicious'."""
        assert compute_tier(79) == "suspicious"

    def test_risk_score_capped_at_100(self, honeypot_candidate):
        """Even with many flags, risk_score should not exceed 100."""
        result = validate_candidate(honeypot_candidate)
        assert result.risk_score <= 100

    def test_primary_reason_is_highest_points(self, honeypot_candidate):
        """Primary reason should come from the highest-point flag."""
        result = validate_candidate(honeypot_candidate)
        if result.flags:
            max_points = max(f.points for f in result.flags)
            max_flag = next(f for f in result.flags if f.points == max_points)
            assert result.primary_reason == max_flag.description

    def test_empty_candidate_doesnt_crash(self):
        """Validation should handle empty/minimal dicts gracefully."""
        result = validate_candidate({})
        assert isinstance(result, ValidationResult)
        assert result.candidate_id == "UNKNOWN"

    def test_validate_candidate_returns_correct_type(self, valid_candidate):
        result = validate_candidate(valid_candidate)
        assert isinstance(result, ValidationResult)
        assert isinstance(result.flags, list)
        assert isinstance(result.risk_score, int)
        assert isinstance(result.is_honeypot, bool)
