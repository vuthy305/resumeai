"""Evaluate the MATCHING stage against baselines on real labelled resume-job pairs.

Dataset: cnamuangtoun/resume-job-description-fit (Hugging Face)
    8,000 (resume, job description, label) pairs; label = No Fit / Potential Fit / Good Fit
    train 6,241 / test 1,759. Everything is tuned on train and reported on test only.

Methods compared (all get exactly the same resume text + job description text):
    1. TF-IDF + cosine similarity          classic IR baseline, unsupervised
    2. TF-IDF + Logistic Regression        classic supervised ML baseline
    3. Structured AI Match Score (ours)    skills 50% / experience 25% / education 15% / preferred 10%
    4. Hybrid (ours + TF-IDF)              experiment: weight chosen on train
    5. Sentence embeddings (optional)      deep-learning baseline, --embeddings (all-MiniLM-L6-v2)

Metrics (test split):
    Spearman rho   does the score order candidates like the labels do? (No=0 < Potential=1 < Good=2)
    AUC Good       can the score separate Good Fit from the rest?
    AUC Fit        can the score separate Potential/Good Fit from No Fit?
    Macro-F1       3-class accuracy after cutting the score with 2 thresholds tuned on train
    NDCG@5         per job ad: are the best-fitting resumes at the top of the ranking?

Run:  python scripts/evaluate_matching.py [--embeddings] [--limit N]
Out:  results/matching_results.csv, results/matching_comparison.png, results/matching_predictions.csv
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DIR, RESULTS_DIR  # noqa: E402
from src.extraction.profile_builder import ProfileBuilder  # noqa: E402
from src.extraction.reporting import bar_plot, save_rows  # noqa: E402
from src.matching.jd_parser import parse_job_text  # noqa: E402
from src.matching.matcher import match_candidate  # noqa: E402

DATASET = "cnamuangtoun/resume-job-description-fit"
LOCAL = RAW_DIR / "resume_jd_fit"
LABELS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
NAMES = ["No Fit", "Potential Fit", "Good Fit"]


# ------------------------------------------------------------------ data
def load_pairs(limit=None):
    from datasets import load_dataset, load_from_disk
    arrow = LOCAL / "arrow"
    if (arrow / "dataset_dict.json").exists():
        print(f"Loading LOCAL copy {arrow}")
        ds = load_from_disk(str(arrow))
    else:
        print(f"Downloading {DATASET} (saved to {arrow} for next time)")
        ds = load_dataset(DATASET)  # default HF cache keeps Windows paths short
        ds.save_to_disk(str(arrow))
    cols = ds["train"].column_names
    rc = next(c for c in cols if "resume" in c.lower())
    jc = next(c for c in cols if "job" in c.lower())
    lc = next(c for c in cols if "label" in c.lower())
    out = {}
    for split in ("train", "test"):
        part = ds[split]
        n = min(limit, len(part)) if limit else len(part)
        rows = [part[i] for i in range(n)]
        out[split] = {"resume": [r[rc] or "" for r in rows], "job": [r[jc] or "" for r in rows],
                      "y": np.array([LABELS[str(r[lc]).strip()] for r in rows])}
        print(f"  {split}: {n} pairs | " + ", ".join(f"{k} {int((out[split]['y'] == v).sum())}" for k, v in LABELS.items()))
    return out


# ------------------------------------------------------------------ methods
def tfidf_fit(train):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=50000)
    vec.fit(list(dict.fromkeys(train["resume"] + train["job"])))
    return vec


def tfidf_cosine(vec, split):
    r, j = vec.transform(split["resume"]), vec.transform(split["job"])  # rows are L2-normalised
    return np.asarray(r.multiply(j).sum(axis=1)).ravel()


def tfidf_features(vec, split):
    from scipy.sparse import hstack
    r, j = vec.transform(split["resume"]), vec.transform(split["job"])
    cos = np.asarray(r.multiply(j).sum(axis=1))
    return hstack([r, j, r.multiply(j), cos]).tocsr()


def structured_scores(split, cache):
    pb = ProfileBuilder()
    out = []
    for resume, job in zip(split["resume"], split["job"]):
        if resume not in cache["cv"]:
            cache["cv"][resume] = pb.build(resume)
        if job not in cache["jd"]:
            cache["jd"][job] = parse_job_text(job)
        out.append(match_candidate(cache["cv"][resume], cache["jd"][job])["match_score"] / 100.0)
    return np.array(out)


def embedding_scores(model, split, cache):
    texts = [t for t in dict.fromkeys(split["resume"] + split["job"]) if t not in cache]
    if texts:
        for t, e in zip(texts, model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)):
            cache[t] = e
    return np.array([float(np.dot(cache[r], cache[j])) for r, j in zip(split["resume"], split["job"])])


# ------------------------------------------------------------------ metrics
def tune_thresholds(score, y):
    """Two cut points (No | Potential | Good) chosen on TRAIN to maximise macro-F1."""
    from sklearn.metrics import f1_score
    qs = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 49)))
    best = (-1, qs[0], qs[-1])
    for i, a in enumerate(qs):
        for b in qs[i:]:
            f = f1_score(y, np.digitize(score, [a, b]), average="macro")
            if f > best[0]:
                best = (f, a, b)
    return best[1], best[2]


def ndcg_at_k(score, y, jobs, k=5):
    groups = defaultdict(list)
    for s, t, j in zip(score, y, jobs):
        groups[j].append((s, t))
    vals = []
    for items in groups.values():
        if len(items) < 2 or len({t for _, t in items}) < 2:
            continue  # nothing to rank
        gains = [t for _, t in sorted(items, key=lambda x: -x[0])][:k]
        ideal = sorted((t for _, t in items), reverse=True)[:k]
        dcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
        vals.append(dcg / idcg if idcg else 0)
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def evaluate(name, kind, s_train, s_test, y_train, y_test, jobs_test, pred_test=None):
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score, roc_auc_score
    if pred_test is None:
        a, b = tune_thresholds(s_train, y_train)
        pred_test = np.digitize(s_test, [a, b])
    ndcg, n_groups = ndcg_at_k(s_test, y_test, jobs_test)
    return {"method": name, "type": kind,
            "spearman": round(float(spearmanr(s_test, y_test).correlation) * 100, 2),
            "auc_good": round(roc_auc_score(y_test == 2, s_test) * 100, 2),
            "auc_fit": round(roc_auc_score(y_test >= 1, s_test) * 100, 2),
            "macro_f1": round(f1_score(y_test, pred_test, average="macro") * 100, 2),
            "ndcg@5": round(ndcg * 100, 2), "ndcg_jobs": n_groups}, pred_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true", help="also run a sentence-embedding baseline (downloads ~90 MB)")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N pairs per split (quick test)")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression

    data = load_pairs(args.limit)
    tr, te = data["train"], data["test"]
    rows, preds, scores = [], {}, {}

    print("\n[1] TF-IDF + cosine similarity")
    vec = tfidf_fit(tr)
    cos_tr, cos_te = tfidf_cosine(vec, tr), tfidf_cosine(vec, te)
    r, preds["tfidf_cosine"] = evaluate("TF-IDF cosine", "baseline (unsupervised)", cos_tr, cos_te, tr["y"], te["y"], te["job"])
    rows.append(r); scores["tfidf_cosine"] = cos_te

    print("[2] TF-IDF + Logistic Regression")
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(tfidf_features(vec, tr), tr["y"])
    p_tr, p_te = clf.predict_proba(tfidf_features(vec, tr)), clf.predict_proba(tfidf_features(vec, te))
    lr_tr, lr_te = p_tr[:, 2] + 0.5 * p_tr[:, 1], p_te[:, 2] + 0.5 * p_te[:, 1]
    r, preds["tfidf_logreg"] = evaluate("TF-IDF + LogReg", "baseline (supervised)", lr_tr, lr_te, tr["y"], te["y"], te["job"],
                                       pred_test=p_te.argmax(1))
    rows.append(r); scores["tfidf_logreg"] = lr_te

    print("[3] Structured AI Match Score (ours)")
    cache = {"cv": {}, "jd": {}}
    st_tr, st_te = structured_scores(tr, cache), structured_scores(te, cache)
    r, preds["structured"] = evaluate("AI Match Score (ours)", "ours (explainable)", st_tr, st_te, tr["y"], te["y"], te["job"])
    rows.append(r); scores["structured"] = st_te

    print("[4] Hybrid: alpha x ours + (1 - alpha) x TF-IDF cosine (alpha tuned on train)")
    from scipy.stats import spearmanr
    rank = lambda v, ref: np.searchsorted(np.sort(ref), v) / len(ref)  # noqa: E731  (percentile scale)
    best = max(np.linspace(0, 1, 11), key=lambda a: spearmanr(a * rank(st_tr, st_tr) + (1 - a) * rank(cos_tr, cos_tr), tr["y"]).correlation)
    hy_tr = best * rank(st_tr, st_tr) + (1 - best) * rank(cos_tr, cos_tr)
    hy_te = best * rank(st_te, st_tr) + (1 - best) * rank(cos_te, cos_tr)
    r, preds["hybrid"] = evaluate(f"Hybrid (alpha={best:.1f})", "experiment", hy_tr, hy_te, tr["y"], te["y"], te["job"])
    rows.append(r); scores["hybrid"] = hy_te

    if args.embeddings:
        print(f"[5] Sentence embeddings: {args.embedding_model}")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.embedding_model)
        ecache = {}
        em_tr, em_te = embedding_scores(model, tr, ecache), embedding_scores(model, te, ecache)
        r, preds["embeddings"] = evaluate("Sentence embeddings", "baseline (deep learning)", em_tr, em_te, tr["y"], te["y"], te["job"])
        rows.append(r); scores["embeddings"] = em_te

    # ------------------------------------------------------------ report
    n_skills = [len(cache["jd"][j]["required_skills"]) for j in dict.fromkeys(te["job"])]
    coverage = {"test_jobs": len(n_skills), "jobs_with_0_dictionary_skills": int(sum(n == 0 for n in n_skills)),
                "avg_skills_per_job": round(float(np.mean(n_skills)), 1) if n_skills else 0}
    print(f"\n{'Method':<26}{'Spearman':>9}{'AUC Good':>10}{'AUC Fit':>9}{'Macro-F1':>10}{'NDCG@5':>8}")
    for r in rows:
        print(f"{r['method']:<26}{r['spearman']:>9.2f}{r['auc_good']:>10.2f}{r['auc_fit']:>9.2f}{r['macro_f1']:>10.2f}{r['ndcg@5']:>8.2f}")
    print(f"\nSkill-dictionary coverage on test job ads: {coverage}")
    print("All metrics x100. Chance level: Spearman 0, AUC 50.")

    RESULTS_DIR.mkdir(exist_ok=True)
    save_rows(rows, RESULTS_DIR / "matching_results.csv",
              extra={"dataset": DATASET, "n_train": len(tr["y"]), "n_test": len(te["y"])})
    (RESULTS_DIR / "matching_dataset_notes.json").write_text(json.dumps(coverage, indent=2))
    metrics = ["spearman", "auc_good", "auc_fit", "macro_f1", "ndcg@5"]
    bar_plot({r["method"]: {m: max(r[m], 0) for m in metrics} for r in rows},
             RESULTS_DIR / "matching_comparison.png", "Resume-job matching on the test split", ylabel="Score (x100)")
    pred_rows = [{"job_id": hashlib.md5(j.encode()).hexdigest()[:8], "label": NAMES[y], **{k: round(float(v[i]), 4) for k, v in scores.items()},
                  **{f"pred_{k}": NAMES[int(v[i])] for k, v in preds.items()}} for i, (j, y) in enumerate(zip(te["job"], te["y"]))]
    save_rows(pred_rows, RESULTS_DIR / "matching_predictions.csv")


if __name__ == "__main__":
    main()
