"""BASELINE: PaddleOCR words/boxes -> regex + keyword + layout rules -> BIO tags.
Deterministic and produces the same 5 entity types as LayoutLMv3."""
import re

from src.extraction.entities import bio_to_entities, entities_to_fields
from src.utils.text_utils import EMAIL_RE, find_emails, find_phones, group_lines, is_phone

SECTION_KEYWORDS = {
    "EDUCATION", "EXPERIENCE", "SKILLS", "WORK EXPERIENCE", "EMPLOYMENT", "EMPLOYMENT HISTORY",
    "PROJECTS", "CERTIFICATIONS", "CERTIFICATES", "CONTACT", "CONTACT INFORMATION", "PROFILE",
    "SUMMARY", "PROFESSIONAL SUMMARY", "OBJECTIVE", "CAREER OBJECTIVE", "PROFESSIONAL EXPERIENCE",
    "WORK HISTORY", "TECHNICAL SKILLS", "KEY SKILLS", "CORE COMPETENCIES", "LANGUAGES", "INTERESTS",
    "HOBBIES", "REFERENCES", "AWARDS", "ACHIEVEMENTS", "ACCOMPLISHMENTS", "QUALIFICATIONS",
    "PERSONAL DETAILS", "PERSONAL INFORMATION", "PUBLICATIONS", "ACTIVITIES", "VOLUNTEER EXPERIENCE",
    "TRAINING", "COURSES", "LEADERSHIP", "ABOUT ME", "STRENGTHS", "EXTRACURRICULAR ACTIVITIES",
}
HEADER_TAILS = {"EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS", "CERTIFICATIONS", "SUMMARY",
                "OBJECTIVE", "PROFILE", "LANGUAGES", "INTERESTS", "REFERENCES", "AWARDS",
                "ACHIEVEMENTS", "QUALIFICATIONS", "HISTORY", "INFORMATION", "DETAILS", "ACTIVITIES"}

ADDRESS_RE = re.compile(
    r"\b(street|st\.|road|rd\.?|avenue|ave\.?|boulevard|blvd|lane|drive|highway|suite|apt\.?|"
    r"apartment|floor|village|district|sangkat|khan|province|city|county|state)\b"
    r"|\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b|\b\d{5}(?:-\d{4})?\b|^\s*(address|location)\s*:", re.IGNORECASE)
CITY_COUNTRY_RE = re.compile(r"^[A-Z][A-Za-z.' ]+,\s*[A-Z][A-Za-z.' ]+$")
NAME_WORD_RE = re.compile(r"^[A-Z][A-Za-z.'-]*$")
SEPARATORS = {"|", "•", "·", "-", "–", ",", "/", ":"}


def header_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z& ]", " ", text)).strip().upper()


def is_section_header(text: str) -> bool:
    key = header_key(text)
    n_words = len(key.split())
    if not key or n_words > 4:
        return False
    if key in SECTION_KEYWORDS:
        return True
    stripped = text.strip().rstrip(":")
    looks_header = stripped.isupper() or stripped.istitle()
    return looks_header and key.split()[-1] in HEADER_TAILS


class RuleBasedExtractor:
    def __init__(self, name_top: int = 300, address_top: int = 400, max_name_lines: int = 6):
        self.name_top = name_top          # y in 0-1000 units
        self.address_top = address_top
        self.max_name_lines = max_name_lines

    # ---------- word-level tagging (used for evaluation and in the app) ----------
    def tag(self, words: list[str], boxes: list[list[int]]) -> list[str]:
        n = len(words)
        labels = ["O"] * n
        for i, w in enumerate(words):
            if EMAIL_RE.search(w):
                labels[i] = "B-EMAIL"
        self._tag_phones(words, labels)
        lines = group_lines(boxes)
        for line in lines:
            if is_section_header(" ".join(words[i] for i in line)):
                self._set_span(labels, line, "SECTIONHEADER")
        self._tag_name(words, boxes, lines, labels)
        self._tag_address(words, boxes, lines, labels)
        return labels

    @staticmethod
    def _set_span(labels, idxs, etype):
        idxs = [i for i in idxs if labels[i] == "O"]
        for k, i in enumerate(idxs):
            labels[i] = ("B-" if k == 0 else "I-") + etype

    def _tag_phones(self, words, labels, max_window: int = 5):
        i, n = 0, len(words)
        while i < n:
            found = 0
            for size in range(min(max_window, n - i), 0, -1):
                chunk = words[i:i + size]
                s = " ".join(chunk)
                if all(re.fullmatch(r"[+()\d.\-/]+", w) for w in chunk) and is_phone(s.strip("()")):
                    found = size
                    break
            if found:
                self._set_span(labels, list(range(i, i + found)), "PHONE")
                i += found
            else:
                # tokens like "Tel:+85512345678"
                if find_phones(words[i]) and labels[i] == "O" and not re.fullmatch(r"(19|20)\d{2}", words[i]):
                    labels[i] = "B-PHONE"
                i += 1

    def _tag_name(self, words, boxes, lines, labels):
        for line in lines[: self.max_name_lines]:
            if boxes[line[0]][1] > self.name_top:
                break
            toks = [i for i in line if labels[i] == "O"]
            if toks and words[toks[0]].lower().rstrip(":") == "name":
                toks = toks[1:]
            ws = [words[i] for i in toks]
            if 2 <= len(ws) <= 4 and all(NAME_WORD_RE.match(w) for w in ws) \
                    and not is_section_header(" ".join(ws)):
                self._set_span(labels, toks, "NAME")
                return

    def _tag_address(self, words, boxes, lines, labels):
        for line in lines:
            if boxes[line[0]][1] > self.address_top:
                break
            text = " ".join(words[i] for i in line)
            if not (ADDRESS_RE.search(text) or (len(line) <= 5 and CITY_COUNTRY_RE.match(text))):
                continue
            toks = [i for i in line if labels[i] == "O" and words[i] not in SEPARATORS]
            if toks and words[toks[0]].lower().rstrip(":") in ("address", "location", "addr"):
                toks = toks[1:]
            if toks:
                self._set_span(labels, toks, "ADDRESS")

    # ---------- field extraction ----------
    def extract(self, words, boxes, page: int = 1) -> dict:
        labels = self.tag(words, boxes)
        fields = entities_to_fields(bio_to_entities(words, labels, page))
        fields["labels"] = labels
        return fields

    def extract_from_text(self, text: str) -> dict:
        """For DOCX (no boxes): line-based version of the same rules."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        fields = {"name": None, "email": None, "phone": None, "address": None,
                  "section_headers": [], "evidence": []}
        emails, phones = find_emails(text), find_phones(text)
        fields["email"] = emails[0] if emails else None
        fields["phone"] = phones[0] if phones else None
        for l in lines[: self.max_name_lines]:
            ws = re.sub(r"^name\s*:\s*", "", l, flags=re.I).split()
            if 2 <= len(ws) <= 4 and all(NAME_WORD_RE.match(w) for w in ws) and not is_section_header(l):
                fields["name"] = " ".join(ws)
                break
        for l in lines[:10]:
            if (ADDRESS_RE.search(l) or (len(l.split()) <= 5 and CITY_COUNTRY_RE.match(l))) \
                    and not EMAIL_RE.search(l) and not is_section_header(l):
                fields["address"] = re.sub(r"^(address|location)\s*:\s*", "", l, flags=re.I)
                break
        for l in lines:
            if is_section_header(l) and l.strip(" :") not in fields["section_headers"]:
                fields["section_headers"].append(l.strip(" :"))
        for k in ("name", "email", "phone", "address"):
            if fields[k]:
                fields["evidence"].append({"text": fields[k], "entity": k.upper(), "page": 1})
        return fields
