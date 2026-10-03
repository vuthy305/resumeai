from src.matching.ranking import rank_applications


def test_ranking_sorted_descending_with_pending_last():
    items = [{"candidate_id": 1, "candidate_name": "B", "match_score": 70},
             {"candidate_id": 2, "candidate_name": "A", "match_score": None},
             {"candidate_id": 3, "candidate_name": "C", "match_score": 91.2},
             {"candidate_id": 4, "candidate_name": "A", "match_score": 70}]
    r = rank_applications(items)
    assert [(x["rank"], x["candidate_id"]) for x in r] == [(1, 3), (2, 4), (3, 1), (None, 2)]


def test_ranking_empty():
    assert rank_applications([]) == []
