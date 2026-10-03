from src.matching.matcher import match_candidate
from src.matching.scoring import combine, education_score, experience_score
from src.utils.skills import get_normalizer

JOB = {"required_skills": ["Python", "PostgreSQL", "REST API"], "preferred_skills": ["Docker", "FastAPI"],
       "minimum_experience_years": 2, "education_requirement": "Bachelor's degree in Computer Science"}


def test_skill_normalization():
    n = get_normalizer()
    assert n.normalize("postgres") == "PostgreSQL"
    assert n.normalize("PostgreSQL Database") == "PostgreSQL"
    assert n.normalize("python 3") == "Python"
    assert n.normalize("Kotlin") == "Kotlin"                       # unknown kept as-is
    assert n.find_in_text("Worked with Python 3 and postgres") == ["Python", "PostgreSQL"]


def test_no_over_normalization_false_positives():
    n = get_normalizer()
    assert n.find_in_text("the rest of the team") == []            # 'rest' is not REST API
    assert n.find_in_text("I excel at problem solving") == []             # 'excel' is not Excel
    assert n.find_in_text("JavaScript") == ["JavaScript"]          # not Java
    assert n.find_in_text("R&D department, Go to market") == []    # bare R / Go not matched


def test_component_scores():
    assert experience_score(3, 2) == 100 and experience_score(1, 2) == 50 and experience_score(None, 0) == 100
    assert education_score(3, 3) == 100 and education_score(2, 3) == 50 and education_score(0, 3) == 0
    assert combine({"required_skills": 100, "experience": 100, "education": 100, "preferred_skills": 0}) == 90.0


def test_match_example_from_spec():
    profile = {"skills": ["Python", "postgres", "Docker"], "total_experience_years": 3,
               "highest_degree_level": 3, "experience_years_source": "date_ranges"}
    m = match_candidate(profile, JOB)
    assert m["required"] == {"matched": ["Python", "PostgreSQL"], "missing": ["REST API"]}
    assert m["preferred"]["matched"] == ["Docker"]
    assert m["experience"]["met"] and m["education"]["met"]
    # 66.67*0.5 + 100*0.25 + 100*0.15 + 50*0.10 = 78.33
    assert m["match_score"] == 78.34 or abs(m["match_score"] - 78.33) < 0.02
    assert m["explanation"][0] == "Required skills: 2/3 matched"


def test_custom_weights_and_empty_profile():
    m = match_candidate({}, JOB, weights={"required_skills": 1, "experience": 0, "education": 0, "preferred_skills": 0})
    assert m["match_score"] == 0.0 and m["required"]["missing"] == ["Python", "PostgreSQL", "REST API"]
