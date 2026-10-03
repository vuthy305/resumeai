"""Rank applications for one job by AI Match Score (descending). Unanalysed ones go last."""


def rank_applications(items: list[dict], score_key: str = "match_score") -> list[dict]:
    scored = [i for i in items if i.get(score_key) is not None]
    pending = [i for i in items if i.get(score_key) is None]
    scored.sort(key=lambda i: (-i[score_key], str(i.get("candidate_name") or ""), i.get("candidate_id") or 0))
    out = []
    for r, item in enumerate(scored, 1):
        out.append({**item, "rank": r})
    for item in pending:
        out.append({**item, "rank": None})
    return out
