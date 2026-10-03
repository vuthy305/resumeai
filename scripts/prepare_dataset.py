"""Download/inspect/validate Kunling/layoutlm_resume_data and save a model-ready dataset.
    python scripts/prepare_dataset.py [--box-format auto|normalized|pixel] [--no-dedup]
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (DATASET_NAME, DEFAULT_LABELS, LABEL_MAP_PATH, PROCESSED_DATASET_DIR,  # noqa: E402
                        RAW_DATASET_DIR, RAW_DIR, RESULTS_DIR)
from src.utils.text_utils import clip_box_1000, normalize_box  # noqa: E402


def pick(names, candidates):
    return next((c for c in candidates if c in names), None)


def label_names(feature):
    inner = getattr(feature, "feature", None)
    if inner is not None and hasattr(inner, "names"):
        return list(inner.names), "classlabel"
    if inner is not None and getattr(inner, "dtype", "") == "string":
        return None, "string"
    return None, "int"


def to_xyxy(b):
    b = list(b)
    if len(b) == 8:  # 4-point polygon
        xs, ys = b[0::2], b[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]
    return b[:4]


def load_raw():
    """Prefer the local copy made by scripts/download_dataset.py; otherwise download from the Hub."""
    from datasets import load_dataset, load_from_disk
    local = RAW_DATASET_DIR / "arrow"
    if (local / "dataset_dict.json").exists():
        print(f"Loading LOCAL dataset from {local}")
        return load_from_disk(str(local))
    print(f"No local copy at {local}. Downloading {DATASET_NAME} from Hugging Face "
          f"(tip: run python scripts/download_dataset.py once to keep a permanent local copy)")
    return load_dataset(DATASET_NAME, cache_dir=str(RAW_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-format", default="auto", choices=["auto", "normalized", "pixel"])
    ap.add_argument("--no-dedup", action="store_true", help="keep train docs that duplicate test docs")
    args = ap.parse_args()

    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value  # noqa: F401
    raw = load_raw()
    print(raw)
    first = next(iter(raw.values()))
    names = list(first.features)
    cols = {"words": pick(names, ["tokens", "words", "text"]), "boxes": pick(names, ["bboxes", "boxes", "bbox"]),
            "labels": pick(names, ["ner_tags", "labels", "tags", "label"]), "image": pick(names, ["image", "img", "image_path"]),
            "id": pick(names, ["id", "guid", "file_name"])}
    print(f"\nFeature names: {names}\nColumn mapping: {cols}")
    if not all(cols[k] for k in ("words", "boxes", "labels", "image")):
        raise SystemExit("Could not detect tokens/bboxes/labels/image columns - edit cols in prepare_dataset.py")

    lnames, lkind = label_names(first.features[cols["labels"]])
    if lkind == "int" and lnames is None:
        print(f"WARNING: integer labels without names; assuming order {DEFAULT_LABELS}. Verify this!")
        lnames = DEFAULT_LABELS

    if "train" in raw and "test" in raw:
        splits = {"train": raw["train"], "test": raw["test"]}
    else:
        s = first.train_test_split(test_size=0.2, seed=42)
        splits = {"train": s["train"], "test": s["test"]}

    from PIL import Image as PILImage
    stats = {"images_valid": 0, "images_invalid": 0, "boxes_valid": 0, "boxes_fixed": 0, "boxes_invalid": 0,
             "docs_dropped": 0, "empty_tokens_removed": 0, "box_format_detected": Counter()}
    out, label_counter, all_labels = {}, Counter(), set()
    for split, ds in splits.items():
        recs = {"id": [], "words": [], "boxes": [], "labels": [], "image": []}
        for i, ex in enumerate(ds):
            try:
                img = ex[cols["image"]]
                img = PILImage.open(img) if isinstance(img, str) else img
                img = img.convert("RGB")
                w, h = img.size
                stats["images_valid"] += 1
            except Exception:
                stats["images_invalid"] += 1; stats["docs_dropped"] += 1
                continue
            words, boxes, labels = ex[cols["words"]], [to_xyxy(b) for b in ex[cols["boxes"]]], ex[cols["labels"]]
            if not (len(words) == len(boxes) == len(labels)) or not words:
                stats["docs_dropped"] += 1
                continue
            labels = [lnames[l] if lkind != "string" else l for l in labels]
            fmt = args.box_format
            if fmt == "auto":
                m = max(max(b) for b in boxes)
                fmt = "fraction" if m <= 1.0 else ("pixel" if m > 1000 else "normalized")
            stats["box_format_detected"][fmt] += 1
            nw, nb, nl = [], [], []
            for word, b, l in zip(words, boxes, labels):
                if not str(word).strip():
                    stats["empty_tokens_removed"] += 1
                    continue
                if fmt == "pixel":
                    nbx = normalize_box(b, w, h)
                elif fmt == "fraction":
                    nbx = clip_box_1000([v * 1000 for v in b])
                else:
                    nbx = clip_box_1000(b)
                if list(nbx) != [int(round(v)) for v in b] and fmt == "normalized":
                    stats["boxes_fixed"] += 1
                if nbx[2] < nbx[0] or nbx[3] < nbx[1]:
                    stats["boxes_invalid"] += 1
                else:
                    stats["boxes_valid"] += 1
                nw.append(str(word)); nb.append(nbx); nl.append(l)
            if not nw:
                stats["docs_dropped"] += 1
                continue
            label_counter.update(nl); all_labels.update(nl)
            recs["id"].append(str(ex[cols["id"]]) if cols["id"] else f"{split}-{i}")
            recs["words"].append(nw); recs["boxes"].append(nb); recs["labels"].append(nl); recs["image"].append(img)
        out[split] = recs

    # duplicate checking (exact token sequence)
    h = {s: [hashlib.md5(" ".join(r).encode()).hexdigest() for r in out[s]["words"]] for s in out}
    dup_train = len(h["train"]) - len(set(h["train"]))
    leak = set(h["train"]) & set(h["test"])
    print(f"\nDuplicates inside train: {dup_train} | train docs also in test: {len(leak)}")
    if leak and not args.no_dedup:
        keep = [i for i, x in enumerate(h["train"]) if x not in leak]
        out["train"] = {k: [v[i] for i in keep] for k, v in out["train"].items()}
        print(f"Removed {len(h['train']) - len(keep)} leaking train docs (use --no-dedup to keep).")

    label_list = ["O"] + sorted(l for l in all_labels if l != "O")
    for l in (lnames or []):
        if l not in label_list:
            label_list.append(l)
    feats = Features({"id": Value("string"), "words": Sequence(Value("string")),
                      "boxes": Sequence(Sequence(Value("int32"))), "labels": Sequence(Value("string")),
                      "image": Image()})
    dd = DatasetDict({s: Dataset.from_dict(out[s], features=feats) for s in out})
    dd.save_to_disk(str(PROCESSED_DATASET_DIR))
    LABEL_MAP_PATH.write_text(json.dumps({"labels": label_list, "label2id": {l: i for i, l in enumerate(label_list)},
                                          "source_label_names": lnames}, indent=2))

    print("\n=================================")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Total: {sum(len(dd[s]) for s in dd)}")
    print(f"Train: {len(dd['train'])}")
    print(f"Test: {len(dd['test'])}")
    print(f"\nImages:\nValid: {stats['images_valid']}\nInvalid: {stats['images_invalid']}")
    print("\nLabels:")
    for l in label_list:
        print(f"  {l:<18} {label_counter[l]}")
    print(f"\nBounding Boxes:\nValid: {stats['boxes_valid']}\nInvalid: {stats['boxes_invalid']}")
    print(f"Box format detected: {dict(stats['box_format_detected'])} | clipped to 0-1000: {stats['boxes_fixed']}")
    print(f"Docs dropped: {stats['docs_dropped']} | empty tokens removed: {stats['empty_tokens_removed']}")
    print("=================================")
    print(f"Saved model-ready dataset -> {PROCESSED_DATASET_DIR}\nLabel map -> {LABEL_MAP_PATH}")
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(
        {**{k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()},
         "train": len(dd["train"]), "test": len(dd["test"]), "label_counts": dict(label_counter),
         "duplicates_in_train": dup_train, "train_test_overlap": len(leak)}, indent=2))


if __name__ == "__main__":
    main()
