"""Download Kunling/layoutlm_resume_data once, store it locally, and export an easy-to-view copy.

    python scripts/download_dataset.py                 # download + export viewable files
    python scripts/download_dataset.py --export-only   # rebuild the viewable files from the local copy
    python scripts/download_dataset.py --force         # delete everything and download again

Folder layout (data/raw/layoutlm_resume_data/):
    files/      exact files from https://huggingface.co/datasets/Kunling/layoutlm_resume_data
    arrow/      the same data as a DatasetDict, loaded offline by prepare_dataset.py (used for training)
    info.json   source link, download date, split sizes, columns
    view/       HUMAN-READABLE copy for browsing and demos:
        index.html                 open in a browser: gallery, filter by split, click a CV to see boxes + entities
        images/<split>/<id>.png    original CV page images
        annotated/<split>/<id>.png same images with coloured boxes per label
        annotations/<split>/<id>.json   words, boxes (0-1000), BIO labels and grouped entities
        summary.csv                one row per document (words, entity counts, extracted NAME / EMAIL / PHONE)
        label_distribution.csv     label counts per split
"""
import argparse
import csv
import html
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATASET_NAME, DEFAULT_LABELS, RAW_DATASET_DIR  # noqa: E402
from src.extraction.entities import bio_to_entities  # noqa: E402

COLORS = {"NAME": (220, 38, 38), "EMAIL": (37, 99, 235), "PHONE": (22, 163, 74),
          "ADDRESS": (234, 88, 12), "SECTIONHEADER": (147, 51, 234)}
ENTITIES = list(COLORS)


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 ** 2


def pick(names, candidates):
    return next((c for c in candidates if c in names), None)


def to_xyxy(b):
    b = list(b)
    if len(b) == 8:
        xs, ys = b[0::2], b[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]
    return b[:4]


def label_decoder(feature):
    inner = getattr(feature, "feature", None)
    if inner is not None and hasattr(inner, "names"):
        return lambda l: inner.names[l]
    if inner is not None and getattr(inner, "dtype", "") == "string":
        return lambda l: l
    print(f"WARNING: integer labels without names; assuming order {DEFAULT_LABELS}")
    return lambda l: DEFAULT_LABELS[l]


def to_pixels(boxes, w, h):
    """Boxes in the dataset may be 0-1000, pixels or 0-1 fractions. Return pixel boxes for drawing."""
    m = max((max(b) for b in boxes), default=0)
    if m <= 1.0:
        return [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes], "fraction"
    if m > 1000:
        return boxes, "pixel"
    return [[b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000] for b in boxes], "normalized_0_1000"


def draw(img, pix_boxes, labels):
    from PIL import ImageDraw
    out = img.copy()
    d = ImageDraw.Draw(out, "RGBA")
    width = max(2, round(max(out.size) / 500))
    for b, l in zip(pix_boxes, labels):
        ent = l.split("-", 1)[-1] if l != "O" else None
        if ent in COLORS:
            d.rectangle(b, outline=COLORS[ent] + (255,), fill=COLORS[ent] + (40,), width=width)
        else:
            d.rectangle(b, outline=(160, 160, 160, 140), width=1)
    return out


def export_view(ds, view_dir: Path, limit: int | None = None):
    from PIL import Image as PILImage
    if view_dir.exists():
        shutil.rmtree(view_dir)
    first = next(iter(ds.values()))
    names = list(first.features)
    cols = {"words": pick(names, ["tokens", "words", "text"]), "boxes": pick(names, ["bboxes", "boxes", "bbox"]),
            "labels": pick(names, ["ner_tags", "labels", "tags", "label"]),
            "image": pick(names, ["image", "img", "image_path"]), "id": pick(names, ["id", "guid", "file_name"])}
    if not all(cols[k] for k in ("words", "boxes", "labels", "image")):
        raise SystemExit(f"Could not detect columns in {names}. Edit cols in download_dataset.py")
    decode = label_decoder(first.features[cols["labels"]])

    rows, dist, docs_js = [], {}, []
    for split, part in ds.items():
        dist[split] = Counter()
        for sub in ("images", "annotated", "annotations"):
            (view_dir / sub / split).mkdir(parents=True, exist_ok=True)
        n = len(part) if not limit else min(limit, len(part))
        for i in range(n):
            ex = part[i]
            doc_id = str(ex[cols["id"]]) if cols["id"] else f"{i:04d}"
            doc_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in doc_id) or f"{i:04d}"
            img = ex[cols["image"]]
            img = (PILImage.open(img) if isinstance(img, str) else img).convert("RGB")
            words = [str(w) for w in ex[cols["words"]]]
            boxes = [to_xyxy(b) for b in ex[cols["boxes"]]]
            labels = [decode(l) for l in ex[cols["labels"]]]
            dist[split].update(labels)
            pix, fmt = to_pixels(boxes, *img.size)
            entities = bio_to_entities(words, labels)
            for e in entities:
                e.pop("page", None)

            img.save(view_dir / "images" / split / f"{doc_id}.png")
            draw(img, pix, labels).save(view_dir / "annotated" / split / f"{doc_id}.png")
            (view_dir / "annotations" / split / f"{doc_id}.json").write_text(json.dumps({
                "id": doc_id, "split": split, "image_size": list(img.size), "box_format": fmt,
                "entities": entities,
                "tokens": [{"word": w, "box": b, "label": l} for w, b, l in zip(words, boxes, labels)],
            }, indent=1, ensure_ascii=False), encoding="utf-8")

            first_of = lambda t: next((e["text"] for e in entities if e["entity"] == t), "")  # noqa: E731
            counts = Counter(e["entity"] for e in entities)
            rows.append({"split": split, "id": doc_id, "num_words": len(words),
                         "image_w": img.size[0], "image_h": img.size[1], "box_format": fmt,
                         **{f"n_{t}": counts.get(t, 0) for t in ENTITIES},
                         "NAME": first_of("NAME"), "EMAIL": first_of("EMAIL"), "PHONE": first_of("PHONE")})
            docs_js.append({"id": doc_id, "split": split, "words": len(words), "entities": entities})
        print(f"  exported {n} {split} documents")

    with open(view_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    all_labels = sorted({l for c in dist.values() for l in c}, key=lambda l: (l != "O", l.split("-")[-1], l))
    with open(view_dir / "label_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", *dist, "total"])
        for l in all_labels:
            w.writerow([l, *[dist[s][l] for s in dist], sum(dist[s][l] for s in dist)])
    write_html(view_dir / "index.html", docs_js, dist)
    return len(rows)


def write_html(path: Path, docs, dist):
    legend = "".join(f'<span class="tag" style="background:rgb{c}">{t}</span>' for t, c in COLORS.items())
    stats = "".join(f"<b>{s}</b>: {sum(1 for d in docs if d['split'] == s)} docs &nbsp; " for s in dist)
    ent_counts = Counter(e["entity"] for d in docs for e in d["entities"])
    ent_stats = " · ".join(f"{t}: {ent_counts.get(t, 0)}" for t in ENTITIES)
    data = json.dumps(docs, ensure_ascii=False).replace("</", "<\\/")
    colors = json.dumps({k: f"rgb{v}" for k, v in COLORS.items()})
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kunling/layoutlm_resume_data - local viewer</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f5f5f4;color:#1c1917}}
header{{padding:16px 24px;background:#1c1917;color:#fff}} header h1{{margin:0 0 6px;font-size:20px}}
header p{{margin:2px 0;font-size:13px;opacity:.85}}
.bar{{padding:10px 24px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#fff;border-bottom:1px solid #ddd}}
.tag{{color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px;padding:20px 24px}}
.card{{background:#fff;border-radius:8px;overflow:hidden;cursor:pointer;box-shadow:0 1px 3px #0002}}
.card img{{width:100%;height:220px;object-fit:cover;object-position:top;display:block}}
.card div{{padding:6px 8px;font-size:12px}}
#modal{{display:none;position:fixed;inset:0;background:#000b;z-index:9;overflow:auto}}
#box{{background:#fff;margin:24px auto;max-width:1300px;border-radius:8px;display:flex;flex-wrap:wrap}}
#box img{{max-width:62%;min-width:300px;flex:1}} #side{{flex:1;min-width:280px;padding:16px;font-size:14px}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #eee;padding:4px;text-align:left;vertical-align:top}}
button,select{{padding:6px 10px}}
</style></head><body>
<header><h1>Kunling/layoutlm_resume_data - local copy</h1>
<p>Source: https://huggingface.co/datasets/Kunling/layoutlm_resume_data · {stats}</p>
<p>Annotated entities: {ent_stats}</p></header>
<div class="bar">Split: <select id="split"><option value="">all</option>{"".join(f'<option>{s}</option>' for s in dist)}</select>
View: <select id="mode"><option value="annotated">annotated boxes</option><option value="images">original</option></select>
<span>{legend}<span class="tag" style="background:#999">O (other words)</span></span></div>
<div class="grid" id="grid"></div>
<div id="modal" onclick="if(event.target.id==='modal')this.style.display='none'"><div id="box"></div></div>
<script>
const DOCS={data}; const COLORS={colors};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
function render(){{
  const sp=document.getElementById('split').value, mode=document.getElementById('mode').value;
  document.getElementById('grid').innerHTML=DOCS.filter(d=>!sp||d.split===sp).map((d,i)=>
   `<div class="card" onclick="openDoc('${{d.split}}','${{d.id}}')"><img loading="lazy" src="${{mode}}/${{d.split}}/${{d.id}}.png">
    <div><b>${{esc(d.id)}}</b> · ${{d.split}}<br>${{esc((d.entities.find(e=>e.entity==='NAME')||{{text:'-'}}).text)}}</div></div>`).join('');
}}
function openDoc(split,id){{
  const d=DOCS.find(x=>x.split===split&&x.id===id), mode=document.getElementById('mode').value;
  const rows=d.entities.map(e=>`<tr><td><span class="tag" style="background:${{COLORS[e.entity]||'#999'}}">${{e.entity}}</span></td><td>${{esc(e.text)}}</td></tr>`).join('');
  document.getElementById('box').innerHTML=`<img src="${{mode}}/${{split}}/${{id}}.png"><div id="side">
   <h3 style="margin-top:0">${{esc(id)}} <small>(${{split}})</small></h3><p>${{d.words}} words · ${{d.entities.length}} labelled entities</p>
   <table><tr><th>Label</th><th>Text</th></tr>${{rows||'<tr><td colspan=2>none</td></tr>'}}</table>
   <p><a href="annotations/${{split}}/${{id}}.json" target="_blank">Open full annotation JSON (words + boxes + labels)</a> ·
   <a href="images/${{split}}/${{id}}.png" target="_blank">original image</a></p>
   <button onclick="document.getElementById('modal').style.display='none'">Close</button></div>`;
  document.getElementById('modal').style.display='block';
}}
document.getElementById('split').onchange=render; document.getElementById('mode').onchange=render; render();
</script></body></html>""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DATASET_NAME, help="Hugging Face dataset id")
    ap.add_argument("--out", default=str(RAW_DATASET_DIR), help="local folder")
    ap.add_argument("--force", action="store_true", help="delete the local copy and download again")
    ap.add_argument("--export-only", action="store_true", help="only rebuild view/ from the existing local copy")
    ap.add_argument("--no-export", action="store_true", help="skip the human-readable view/ export")
    ap.add_argument("--limit", type=int, default=None, help="export only the first N docs per split to view/")
    args = ap.parse_args()

    from datasets import load_dataset, load_from_disk

    out = Path(args.out)
    files_dir, arrow_dir, view_dir = out / "files", out / "arrow", out / "view"
    if args.force and out.exists():
        shutil.rmtree(out)

    if (arrow_dir / "dataset_dict.json").exists():
        print(f"Local copy found at {arrow_dir} (use --force to download again)")
        ds = load_from_disk(str(arrow_dir))
    elif args.export_only:
        raise SystemExit("No local copy yet. Run: python scripts/download_dataset.py")
    else:
        from huggingface_hub import snapshot_download
        print(f"[1/3] Downloading repository files of {args.repo} -> {files_dir}")
        snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(files_dir))
        print(f"[2/3] Building DatasetDict -> {arrow_dir}")
        # This dataset ships a loading script, so the Hub loader is the reliable way to build it.
        # No cache_dir on purpose: the default HF cache (C:\\Users\\<you>\\.cache\\huggingface) keeps
        # file paths short. A cache inside the project folder breaks Windows' 260-character path limit.
        try:
            ds = load_dataset(args.repo, trust_remote_code=True)
            source = "hub loader"
        except Exception as e:  # noqa: BLE001
            print(f"  Hub loader failed ({type(e).__name__}: {e}). Trying the downloaded files directly.")
            ds = load_dataset(str(files_dir))
            source = "local files"
        ds.save_to_disk(str(arrow_dir))
        (out / "info.json").write_text(json.dumps({
            "repo": args.repo, "url": f"https://huggingface.co/datasets/{args.repo}", "built_from": source,
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            "splits": {k: len(v) for k, v in ds.items()},
            "features": {k: str(v) for k, v in next(iter(ds.values())).features.items()}}, indent=2))
    print(ds)

    if not args.no_export:
        print(f"[3/3] Exporting viewable copy -> {view_dir}")
        n = export_view(ds, view_dir, args.limit)
        print(f"  {n} documents exported")
        print(f"\nOpen in your browser:  {(view_dir / 'index.html').resolve()}")
        print(f"Tables:                {view_dir / 'summary.csv'}  and  {view_dir / 'label_distribution.csv'}")
    print(f"\nLocal dataset folder: {out.resolve()} ({dir_size_mb(out):.1f} MB)")
    print("Next: python scripts/prepare_dataset.py   (loads from this local copy, no internet needed)")


if __name__ == "__main__":
    main()