import numpy as np

from scripts.evaluate_matching import ndcg_at_k, tune_thresholds
from src.matching.jd_parser import parse_job_text


def test_jd_parser_extracts_requirements():
    jd = ("Backend Developer. 3+ years of experience with Python and PostgreSQL. "
          "Bachelor's degree required, Master's preferred. 5 years in banking is a plus.")
    r = parse_job_text(jd)
    assert {"Python", "PostgreSQL"} <= set(r["required_skills"])
    assert r["minimum_experience_years"] == 3.0
    assert r["education_requirement"] == "Bachelor's Degree"


def test_jd_parser_empty_text():
    assert parse_job_text("") == {"required_skills": [], "preferred_skills": [],
                                  "minimum_experience_years": 0.0, "education_requirement": ""}


def test_ndcg_perfect_and_reversed():
    y = np.array([2, 1, 0]); jobs = ["a"] * 3
    assert ndcg_at_k(np.array([0.9, 0.5, 0.1]), y, jobs)[0] == 1.0
    assert ndcg_at_k(np.array([0.1, 0.5, 0.9]), y, jobs)[0] < 0.7


def test_thresholds_separate_classes():
    s = np.array([0.1, 0.15, 0.5, 0.55, 0.9, 0.95]); y = np.array([0, 0, 1, 1, 2, 2])
    a, b = tune_thresholds(s, y)
    assert (np.digitize(s, [a, b]) == y).all()