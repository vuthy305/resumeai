"""ABLATION: does layout help?  Same architecture, same data split, same hyper-parameters.
  A) Text-based:     OCR words + 1D sequence order; boxes zeroed, blank image (train + test)
  B) Text + Layout:  OCR words + boxes + page image (the main model)
    python scripts/run_ablation.py --epochs 5            # trains A, evaluates A and B
    python scripts/run_ablation.py --skip-train          # reuse existing A checkpoint
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BEST_MODEL_DIR, ENTITY_TYPES, RESULTS_DIR, TEXT_ONLY_MODEL_DIR  # noqa: E402
from src.extraction.reporting import bar_plot, print_table, save_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="defaults to the main model's epochs")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()
    from src.config import TrainConfig
    from src.extraction.training import evaluate_checkpoint, train

    if not (BEST_MODEL_DIR / "config.json").exists():
        raise SystemExit("Train the main model first: python scripts/train_layoutlmv3.py")
    main_cfg = json.loads((BEST_MODEL_DIR / "train_config.json").read_text())
    if not args.skip_train or not (TEXT_ONLY_MODEL_DIR / "config.json").exists():
        cfg = TrainConfig(**{k: v for k, v in main_cfg.items() if k in TrainConfig.__dataclass_fields__})
        cfg.no_layout, cfg.output_dir = True, str(TEXT_ONLY_MODEL_DIR.parent)
        if args.epochs:
            cfg.epochs = args.epochs
        train(cfg)
    a, _, _ = evaluate_checkpoint(TEXT_ONLY_MODEL_DIR, "test", args.max_length, no_layout=True)
    b, _, _ = evaluate_checkpoint(BEST_MODEL_DIR, "test", args.max_length, no_layout=False)
    print_table(a, "A) Text-based (no boxes, blank image)")
    print_table(b, "B) Text + Layout (boxes + image)")
    rows = []
    for name, r in (("Text-based", a), ("Text + Layout", b)):
        d = {x["entity"]: x for x in r}
        rows.append({"configuration": name, "macro_precision": d["MACRO"]["precision"],
                     "macro_recall": d["MACRO"]["recall"], "macro_f1": d["MACRO"]["f1"],
                     **{f"{e}_f1": d[e]["f1"] for e in ENTITY_TYPES}})
    with open(RESULTS_DIR / "ablation_results.csv", "w", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n{'Configuration':<16}{'Macro F1':>10}")
    for r in rows:
        print(f"{r['configuration']:<16}{r['macro_f1']:>10.2f}")
    save_rows(a, RESULTS_DIR / "ablation_text_only_detail.csv", {"configuration": "Text-based"})
    cats = ENTITY_TYPES + ["MACRO"]
    bar_plot({"Text-based": {c: next(x["f1"] for x in a if x["entity"] == c) for c in cats},
              "Text + Layout": {c: next(x["f1"] for x in b if x["entity"] == c) for c in cats}},
             RESULTS_DIR / "ablation.png", "Ablation: does layout help? (test F1)")


if __name__ == "__main__":
    main()
