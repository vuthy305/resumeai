"""PDF handling with PyMuPDF. Pages are rendered one at a time to keep memory low."""
from pathlib import Path
from typing import Iterator

from PIL import Image

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz

from src.config import PDF_DPI, PDF_MAX_PAGES


class PDFError(ValueError):
    pass


def open_pdf(path: str | Path):
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise PDFError(f"Corrupted or unreadable PDF: {e}") from e
    if doc.page_count == 0:
        doc.close()
        raise PDFError("PDF has no pages.")
    return doc


def page_count(path) -> int:
    with open_pdf(path) as doc:
        return doc.page_count


def iter_pdf_pages(path, dpi: int = PDF_DPI, max_pages: int = PDF_MAX_PAGES) -> Iterator[tuple[int, Image.Image]]:
    """Yield (page_number starting at 1, RGB PIL image). Only one page lives in memory."""
    doc = open_pdf(path)
    try:
        zoom = dpi / 72.0
        for i in range(min(doc.page_count, max_pages)):
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            del pix
            yield i + 1, img
    finally:
        doc.close()


def extract_text_layer_words(path, max_pages: int = PDF_MAX_PAGES) -> list[dict]:
    """Real word boxes from a digital PDF's text layer (no OCR). Used as a fallback when
    PaddleOCR is not installed. Returns one dict per page: words, boxes (0-1000), width, height."""
    from src.utils.text_utils import normalize_box
    pages = []
    doc = open_pdf(path)
    try:
        for i in range(min(doc.page_count, max_pages)):
            page = doc[i]
            w, h = page.rect.width, page.rect.height
            words, boxes = [], []
            # (x0, y0, x1, y1, word, block, line, word_no) sorted in reading order
            for x0, y0, x1, y1, word, *_ in sorted(page.get_text("words"), key=lambda t: (t[5], t[6], t[7])):
                if word.strip():
                    words.append(word)
                    boxes.append(normalize_box([x0, y0, x1, y1], w, h))
            pages.append({"page": i + 1, "words": words, "boxes": boxes, "scores": [1.0] * len(words),
                          "width": w, "height": h})
    finally:
        doc.close()
    return pages


def extract_pdf_text(path, max_pages: int = PDF_MAX_PAGES) -> str:
    doc = open_pdf(path)
    try:
        return "\n".join(doc[i].get_text() for i in range(min(doc.page_count, max_pages)))
    finally:
        doc.close()
