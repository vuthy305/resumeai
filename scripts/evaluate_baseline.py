"""Evaluate BASELINE (rules) on the test split.
  --mode tokens (default): rules run on the dataset's OCR tokens+boxes -> exact entity-span F1 (seqeval).
                           Same input as LayoutLMv3 evaluation, so the comparison is controlled.
  --mode ocr: re-OCR every test image with PaddleOCR -> rules -> string-level F1 vs gold entity strings.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR  # noqa: E402
from src.extraction.baseline import RuleBasedExtractor  # noqa: E402
from src.extraction.entities import bio_to_entities  # noqa: E402
from src.extraction.metrics import seqeval_rows, string_level_rows  # noqa: E402
from src.extraction.reporting import print_table, save_rows  # noqa: E402
from src.extraction.training import load_processed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tokens", "ocr"], default="tokens")
    args = ap.parse_args()
    ds, _ = load_processed()
    test, rules = ds["test"], RuleBasedExtractor()
    if args.mode == "tokens":
        y_true, y_pred = [], []
        for doc in test.remove_columns("image"):
            y_true.append(doc["labels"])
            y_pred.append(rules.tag(doc["words"], doc["boxes"]))
        rows = seqeval_rows(y_true, y_pred)
        out = RESULTS_DIR / "baseline_results.csv"
    else:
        from src.document_processing.ocr import OCRProcessor
        ocr = OCRProcessor()
        gold, pred = [], []
        for i, doc in enumerate(test):
            gold.append(bio_to_entities(doc["words"], doc["labels"]))
            o = ocr.extract(doc["image"])
            pred.append(bio_to_entities(o["words"], rules.tag(o["words"], o["boxes"])))
            print(f"OCR {i + 1}/{len(test)}", end="\r")
        rows = string_level_rows(gold, pred)
        out = RESULTS_DIR / "baseline_ocr_results.csv"
    print_table(rows, f"BASELINE (PaddleOCR + Rules) - mode={args.mode}, {len(test)} test docs")
    save_rows(rows, out, {"method": "PaddleOCR + Rules", "mode": args.mode})


if __name__ == "__main__":
    main()
