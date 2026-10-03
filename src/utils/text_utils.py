"""Shared text/box helpers (no heavy dependencies)."""
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\(?\d[\d\s().-]{6,}\d)")
YEAR_RANGE_RE = re.compile(r"^\(?(19|20)\d{2}\)?\s*[-–—/]\s*\(?(19|20)\d{2}\)?$")
DATE_LIKE_RE = re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def find_emails(text: str) -> list[str]:
    return [m.rstrip(".") for m in EMAIL_RE.findall(text or "")]


def is_phone(candidate: str) -> bool:
    s = candidate.strip()
    digits = re.sub(r"\D", "", s)
    if not 8 <= len(digits) <= 15:
        return False
    if YEAR_RANGE_RE.match(s) or DATE_LIKE_RE.match(s):
        return False
    return True


def find_phones(text: str) -> list[str]:
    out = []
    for m in PHONE_RE.finditer(text or ""):
        s = m.group(0).strip(" .-")
        if is_phone(s):
            out.append(s)
    return out


def normalize_box(box, width: float, height: float) -> list[int]:
    """Pixel [x0,y0,x1,y1] -> LayoutLMv3 0-1000 integer range, clipped and ordered."""
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    def n(v, d):
        return int(max(0, min(1000, round(1000 * v / max(d, 1)))))
    return [n(x0, width), n(y0, height), n(x1, width), n(y1, height)]


def clip_box_1000(box) -> list[int]:
    x0, y0, x1, y1 = [int(max(0, min(1000, round(v)))) for v in box]
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def group_lines(boxes: list[list[int]]) -> list[list[int]]:
    """Group consecutive word indices into visual lines by vertical position.
    Keeps the original word order (the OCR/dataset reading order)."""
    lines, current, cur_yc, cur_h = [], [], None, None
    for i, b in enumerate(boxes):
        yc, h = (b[1] + b[3]) / 2, max(b[3] - b[1], 1)
        if current and abs(yc - cur_yc) <= max(0.6 * cur_h, 2):
            current.append(i)
        else:
            if current:
                lines.append(current)
            current, cur_yc, cur_h = [i], yc, h
    if current:
        lines.append(current)
    return lines


def normalize_for_match(text: str, entity: str = "") -> str:
    """Normalisation used for string-level entity comparison."""
    if entity == "PHONE":
        return re.sub(r"\D", "", text or "")
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip(" .,:;|-")
    return t
