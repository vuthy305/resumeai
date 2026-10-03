"""Entity-level metrics (seqeval, exact span match) and string-level metrics for OCR mode."""
from collections import Counter

from src.config import ENTITY_TYPES
from src.utils.text_utils import normalize_for_match


def seqeval_rows(y_true: list[list[str]], y_pred: list[list[str]]) -> list[dict]:
    from seqeval.metrics import classification_report
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for ent in ENTITY_TYPES:
        r = rep.get(ent, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        rows.append({"entity": ent, "precision": 100 * r["precision"], "recall": 100 * r["recall"],
                     "f1": 100 * r["f1-score"], "support": int(r["support"])})
    return _add_averages(rows, rep.get("micro avg"))


def _add_averages(rows, micro=None):
    present = [r for r in rows if r["support"] > 0]
    n = max(len(present), 1)
    rows.append({"entity": "MACRO", "precision": sum(r["precision"] for r in present) / n,
                 "recall": sum(r["recall"] for r in present) / n, "f1": sum(r["f1"] for r in present) / n,
                 "support": sum(r["support"] for r in present)})
    if micro:
        rows.append({"entity": "MICRO", "precision": 100 * micro["precision"], "recall": 100 * micro["recall"],
                     "f1": 100 * micro["f1-score"], "support": int(micro["support"])})
    for r in rows:
        for k in ("precision", "recall", "f1"):
            r[k] = round(r[k], 2)
    return rows


def string_level_rows(gold_docs: list[list[dict]], pred_docs: list[list[dict]]) -> list[dict]:
    """Each doc: list of {'entity','text'}. Compares normalised strings per document
    (used when predictions come from real PaddleOCR tokens that differ from gold tokens)."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for gold, pred in zip(gold_docs, pred_docs):
        g = Counter((e["entity"], normalize_for_match(e["text"], e["entity"])) for e in gold)
        p = Counter((e["entity"], normalize_for_match(e["text"], e["entity"])) for e in pred)
        for key in set(g) | set(p):
            m = min(g[key], p[key])
            tp[key[0]] += m
            fp[key[0]] += p[key] - m
            fn[key[0]] += g[key] - m
    rows = []
    for ent in ENTITY_TYPES:
        P = tp[ent] / (tp[ent] + fp[ent]) if tp[ent] + fp[ent] else 0.0
        R = tp[ent] / (tp[ent] + fn[ent]) if tp[ent] + fn[ent] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        rows.append({"entity": ent, "precision": 100 * P, "recall": 100 * R, "f1": 100 * F,
                     "support": tp[ent] + fn[ent]})
    return _add_averages(rows)
