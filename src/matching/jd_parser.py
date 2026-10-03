"""Turn a free-text job description into the structured requirements the matcher uses.

Used by the matching evaluation (job ads in the dataset are plain text) and can also pre-fill
the HR "New job" form. Rules only, fully explainable:
  * skills      -> every skill-dictionary term found in the text (treated as required)
  * experience  -> smallest "N years" requirement mentioned (e.g. "3+ years of experience")
  * education   -> lowest degree level mentioned ("Bachelor's required, Master's preferred" -> Bachelor's)
"""
import re

from src.extraction.profile_builder import DEGREE_PATTERNS
from src.utils.skills import get_normalizer

YEARS_RE = re.compile(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)\b", re.I)


def parse_job_text(text: str) -> dict:
    text = text or ""
    skills = get_normalizer().find_in_text(text)
    years = [int(m.group(1)) for m in YEARS_RE.finditer(text) if 0 < int(m.group(1)) <= 15]
    levels = {level: name for level, name, pat in DEGREE_PATTERNS if pat.search(text)}
    edu = levels[min(levels)] if levels else ""
    return {"required_skills": skills, "preferred_skills": [],
            "minimum_experience_years": float(min(years)) if years else 0.0,
            "education_requirement": edu}
