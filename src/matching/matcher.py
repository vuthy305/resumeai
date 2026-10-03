"""Structured CV profile + job requirements -> AI Match Score with a full explanation.
Uses only job-related fields (skills, experience, education). No protected attributes."""
from src.config import DEFAULT_MATCH_WEIGHTS
from src.extraction.profile_builder import degree_level_of
from src.matching.scoring import combine, education_score, experience_score, ratio_score
from src.utils.skills import get_normalizer

LEVEL_NAMES = {0: "Not found", 1: "High School", 2: "Associate Degree / Diploma",
               3: "Bachelor's Degree", 4: "Master's Degree", 5: "PhD / Doctorate"}


def _skill_match(candidate: list[str], required: list[str]):
    norm = get_normalizer()
    cand = {s.lower() for s in norm.normalize_list(candidate)}
    req = norm.normalize_list(required)
    matched = [s for s in req if s.lower() in cand]
    missing = [s for s in req if s.lower() not in cand]
    return matched, missing, ratio_score(len(matched), len(req))


def match_candidate(profile: dict, job: dict, weights: dict | None = None) -> dict:
    weights = weights or DEFAULT_MATCH_WEIGHTS
    skills = profile.get("skills") or []
    req_m, req_x, req_s = _skill_match(skills, job.get("required_skills") or [])
    pref_m, pref_x, pref_s = _skill_match(skills, job.get("preferred_skills") or [])

    years = profile.get("total_experience_years")
    min_years = float(job.get("minimum_experience_years", job.get("minimum_experience", 0)) or 0)
    exp_s = experience_score(years, min_years)

    cand_level = int(profile.get("highest_degree_level") or 0)
    req_level = degree_level_of(job.get("education_requirement") or "")
    edu_s = education_score(cand_level, req_level)

    components = {"required_skills": req_s, "experience": exp_s,
                  "education": edu_s, "preferred_skills": pref_s}
    score = combine(components, weights)
    exp_met = (years or 0) >= min_years
    edu_met = cand_level >= req_level if req_level else True

    explanation = [
        f"Required skills: {len(req_m)}/{len(req_m) + len(req_x)} matched",
        f"Preferred skills: {len(pref_m)}/{len(pref_m) + len(pref_x)} matched",
        f"Experience: {years if years is not None else 0} years vs {min_years:g} years required"
        + ("" if profile.get("experience_years_source") != "not_found" else " (no dates found in CV)"),
        f"Education: {LEVEL_NAMES.get(cand_level)} vs {LEVEL_NAMES.get(req_level) if req_level else 'no requirement'}"
        f" -> {'Matched' if edu_met else 'Not matched'}",
        "Final AI Match Score = " + " + ".join(
            f"{components[k]:.0f}x{weights[k]:.2f}" for k in weights) + f" = {score:.1f}",
    ]
    return {
        "match_score": score,
        "components": components,
        "weights": weights,
        "required": {"matched": req_m, "missing": req_x},
        "preferred": {"matched": pref_m, "missing": pref_x},
        "experience": {"candidate_years": years, "required_years": min_years, "met": exp_met,
                       "source": profile.get("experience_years_source")},
        "education": {"candidate": LEVEL_NAMES.get(cand_level), "candidate_level": cand_level,
                      "required": job.get("education_requirement"), "required_level": req_level,
                      "met": edu_met},
        "explanation": explanation,
    }
