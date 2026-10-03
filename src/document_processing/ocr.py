"""PaddleOCR wrapper returning words + LayoutLMv3-ready boxes (0-1000).

PaddleOCR detects text *lines*. LayoutLMv3 (and the training dataset) work with *words*,
so each OCR line box is split into word boxes proportionally to character offsets inside
the real detected line box. This is an approximation of word position within a detected
line, not an invented box: every word box lies inside a box PaddleOCR actually returned.
Set split_words=False to keep line-level tokens.
"""
import numpy as np
from PIL import Image

from src.config import OCR_USE_GPU
from src.utils.logging_utils import get_logger
from src.utils.text_utils import normalize_box

log = get_logger("ocr")


class OCRError(RuntimeError):
    pass


def poly_to_box(poly) -> list[float]:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]


def split_line_into_words(text: str, box: list[float]) -> list[tuple[str, list[float]]]:
    x0, y0, x1, y1 = box
    n = max(len(text), 1)
    out, pos = [], 0
    for word in text.split():
        start = text.index(word, pos)
        end = start + len(word)
        pos = end
        out.append((word, [x0 + (x1 - x0) * start / n, y0, x0 + (x1 - x0) * end / n, y1]))
    return out


def sort_lines(lines: list[dict]) -> list[dict]:
    """Top-to-bottom, left-to-right with a tolerance band based on median line height."""
    if not lines:
        return lines
    med_h = float(np.median([l["box"][3] - l["box"][1] for l in lines])) or 10.0
    band = max(med_h * 0.5, 1.0)
    return sorted(lines, key=lambda l: (round(((l["box"][1] + l["box"][3]) / 2) / band), l["box"][0]))


def parse_paddle_result(raw) -> list[dict]:
    """Normalise PaddleOCR 2.x (list of [poly, (text, score)]) and 3.x (dict-like with
    rec_texts/rec_scores/rec_polys) outputs into [{'text','score','box'}] in pixels."""
    lines = []
    if not raw:
        return lines
    first = raw[0]
    if first is None:
        return lines
    if hasattr(first, "keys") and "rec_texts" in first.keys():  # PaddleOCR 3.x
        for page in raw:
            polys = page.get("rec_polys")
            if polys is None:
                polys = page.get("dt_polys", [])
            for text, score, poly in zip(page["rec_texts"], page["rec_scores"], polys):
                lines.append({"text": text, "score": float(score), "box": poly_to_box(poly)})
        return lines
    for item in first:  # PaddleOCR 2.x: raw[0] is the first (only) page
        poly, (text, score) = item[0], item[1]
        lines.append({"text": text, "score": float(score), "box": poly_to_box(poly)})
    return lines


class OCRProcessor:
    def __init__(self, lang: str = "en", use_gpu: bool = OCR_USE_GPU, min_score: float = 0.5,
                 split_words: bool = True, engine=None):
        self.min_score = min_score
        self.split_words = split_words
        self.engine = engine if engine is not None else self._load(lang, use_gpu)

    @staticmethod
    def _load(lang: str, use_gpu: bool):
        try:
            import paddleocr
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise OCRError("PaddleOCR is not installed. See README 'Installation'.") from e
        major = int(str(getattr(paddleocr, "__version__", "2")).split(".")[0])
        log.info(f"Loading PaddleOCR {getattr(paddleocr, '__version__', '?')} (gpu={use_gpu})")
        if major >= 3:
            return PaddleOCR(lang=lang, device="gpu" if use_gpu else "cpu",
                             use_doc_orientation_classify=False, use_doc_unwarping=False,
                             use_textline_orientation=False)
        return PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=use_gpu, show_log=False)

    def _run(self, image: Image.Image):
        arr = np.array(image.convert("RGB"))[:, :, ::-1].copy()  # PaddleOCR expects BGR
        if hasattr(self.engine, "predict"):
            return self.engine.predict(arr)
        return self.engine.ocr(arr, cls=False)

    def extract(self, image: Image.Image) -> dict:
        width, height = image.size
        try:
            raw = self._run(image)
        except Exception as e:
            log.exception("PaddleOCR failed")
            raise OCRError(f"OCR failed: {e}") from e
        lines = [l for l in parse_paddle_result(raw) if l["text"].strip() and l["score"] >= self.min_score]
        lines = sort_lines(lines)
        words, boxes, scores, pixel_boxes = [], [], [], []
        for line in lines:
            pieces = split_line_into_words(line["text"], line["box"]) if self.split_words \
                else [(line["text"], line["box"])]
            for word, box in pieces:
                words.append(word)
                pixel_boxes.append([round(v, 1) for v in box])
                boxes.append(normalize_box(box, width, height))
                scores.append(line["score"])
        return {"words": words, "boxes": boxes, "scores": scores, "pixel_boxes": pixel_boxes,
                "lines": [l["text"] for l in lines], "width": width, "height": height}
