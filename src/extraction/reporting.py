"""CSV + plot helpers for evaluation results (only ever fed real computed metrics)."""
import csv
from pathlib import Path

from src.config import ENTITY_TYPES


def save_rows(rows: list[dict], path, extra: dict | None = None):
    rows = [{**(extra or {}), **r} for r in rows]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


def load_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def print_table(rows, title=""):
    if title:
        print(f"\n{title}")
    print(f"{'Entity':<15}{'P':>8}{'R':>8}{'F1':>8}{'Support':>9}")
    for r in rows:
        print(f"{r['entity']:<15}{float(r['precision']):>8.2f}{float(r['recall']):>8.2f}"
              f"{float(r['f1']):>8.2f}{int(float(r['support'])):>9}")


def comparison(baseline_rows, model_rows, out_paths, a="Baseline F1", b="LayoutLMv3 F1"):
    bf = {r["entity"]: float(r["f1"]) for r in baseline_rows}
    mf = {r["entity"]: float(r["f1"]) for r in model_rows}
    rows = [{"entity": e if e != "MACRO" else "Macro F1", a: round(bf.get(e, 0), 2), b: round(mf.get(e, 0), 2),
             "difference": round(mf.get(e, 0) - bf.get(e, 0), 2)} for e in ENTITY_TYPES + ["MACRO"]]
    for p in out_paths:
        save_rows(rows, p)
    print(f"\n{'Entity':<15}{a:>14}{b:>16}")
    for r in rows:
        print(f"{r['entity']:<15}{r[a]:>14.2f}{r[b]:>16.2f}")
    return rows


def bar_plot(groups: dict[str, dict[str, float]], path, title, ylabel="F1 (%)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cats = list(next(iter(groups.values())).keys())
    x = np.arange(len(cats))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(cats)), 4))
    for i, (name, vals) in enumerate(groups.items()):
        bars = ax.bar(x + i * width - 0.4 + width / 2, [vals[c] for c in cats], width, label=name)
        ax.bar_label(bars, fmt="%.1f", fontsize=7)
    ax.set_xticks(x, cats, rotation=15)
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
