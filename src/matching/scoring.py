"""Transparent, configurable scoring. All component scores are 0-100."""
from src.config import DEFAULT_MATCH_WEIGHTS


def ratio_score(matched: int, total: int) -> float:
    return 100.0 if total == 0 else round(100.0 * matched / total, 2)


def experience_score(candidate_years: float | None, required_years: float) -> float:
    if not required_years:
        return 100.0
    return round(100.0 * min((candidate_years or 0.0) / required_years, 1.0), 2)


def education_score(candidate_level: int, required_level: int) -> float:
    """Levels: 1 high school, 2 associate/diploma, 3 bachelor, 4 master, 5 PhD."""
    if not required_level:
        return 100.0
    if candidate_level >= required_level:
        return 100.0
    if candidate_level == required_level - 1:
        return 50.0
    return 0.0


def combine(components: dict, weights: dict | None = None) -> float:
    w = weights or DEFAULT_MATCH_WEIGHTS
    total_w = sum(w.values())
    score = sum(components[k] * w[k] for k in w) / total_w
    return round(score, 2)
