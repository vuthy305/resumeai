"""SUPPLEMENTARY RULE-BASED CV FIELDS.

The supervised dataset only labels NAME/EMAIL/PHONE/ADDRESS/SECTIONHEADER, so the fields
below are NOT learned by LayoutLMv3. They are produced deterministically from the OCR text
(section detection + dictionaries + regex + date parsing). Nothing is invented: a field that
is not found stays [] or None. Protected attributes (age, gender, religion, marital status,
photo, etc.) are intentionally never extracted.
"""
import re
from datetime import date

from src.extraction.baseline import header_key
from src.utils.skills import get_normalizer

SECTION_ALIASES = {
    "skills": ["skills", "technical skills", "key skills", "core competencies", "competencies",
               "skills abilities", "skills & abilities", "technologies", "tools", "technical proficiencies",
               "skill highlights", "areas of expertise"],
    "experience": ["experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "career history", "relevant experience"],
    "education": ["education", "academic background", "education and training", "academic qualifications",
                  "educational background", "qualifications", "education & training"],
    "certifications": ["certifications", "certificates", "certification", "licenses",
                       "licenses & certifications", "licenses and certifications", "training", "courses"],
    "languages": ["languages", "language skills", "language"],
    "summary": ["summary", "professional summary", "profile", "objective", "career objective", "about me"],
    "projects": ["projects", "personal projects", "academic projects"],
    "other": ["interests", "hobbies", "references", "awards", "achievements", "accomplishments",
              "activities", "volunteer experience", "personal details", "personal information",
              "contact", "contact information", "publications", "leadership", "strengths"],
}
_ALIAS_TO_SECTION = {header_key(a): sec for sec, al in SECTION_ALIASES.items() for a in al}

DEGREE_PATTERNS = [
    (5, "PhD / Doctorate", re.compile(r"\bph\.?\s?d\b|\bdoctor(ate)? of\b|\bdoctoral\b", re.I)),
    (4, "Master's Degree", re.compile(r"\bmaster'?s?\b|\bm\.?\s?sc\b|\bmba\b|\bm\.eng\b|\bm\.s\.|\bm\.a\.", re.I)),
    (3, "Bachelor's Degree", re.compile(r"\bbachelor'?s?\b|\bb\.?\s?sc\b|\bb\.s\.|\bb\.a\.|\bb\.?\s?eng\b|\bb\.?\s?tech\b|\bbba\b|\bundergraduate degree\b", re.I)),
    (2, "Associate Degree / Diploma", re.compile(r"\bassociate'?s? (degree|of)\b|\bdiploma\b|\bhnd\b", re.I)),
    (1, "High School", re.compile(r"\bhigh school\b|\bsecondary school\b|\bged\b", re.I)),
]

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
DATE_RE = rf"(?:{MONTH_RE}\s*,?\s*(?:19|20)\d{{2}}|\d{{1,2}}\s*[/.-]\s*(?:19|20)\d{{2}}|(?:19|20)\d{{2}})"
RANGE_RE = re.compile(rf"({DATE_RE})\s*(?:-|–|—|to|until|till)\s*({DATE_RE}|present|current|now|today|date)", re.I)
EXPLICIT_YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:professional\s+|work\s+|working\s+|industry\s+|relevant\s+)?experience", re.I)

TITLE_RE = re.compile(
    r"\b((?:senior|junior|lead|principal|chief|assistant|associate|head)?\s*(?:[A-Z][A-Za-z/&+.-]*\s){0,3}"
    r"(?:engineer|developer|programmer|analyst|officer|manager|specialist|consultant|administrator|"
    r"technician|intern|accountant|coordinator|architect|scientist|designer|teller|auditor|executive|"
    r"supervisor|director|representative|support))\b", re.I)
COMPANY_RE = re.compile(
    r"\b((?:[A-Z][\w&.'-]*\s){1,5}(?:Co\.,?\s*Ltd|Inc|Ltd|LLC|Corp|Corporation|Company|Co|Bank|Group|Plc|PLC|Limited|"
    r"Technologies|Technology|Solutions|Systems|Microfinance|Consulting|Enterprises|Holdings|University)\b\.?)")
AT_COMPANY_RE = re.compile(r"\bat\s+((?:[A-Z][\w&.'-]*\s?){1,5})")
CERT_RE = re.compile(r"\bcertified\b|\bcertificate\b|\bcertification\b|\b(PMP|CFA|ACCA|CPA|CCNA|CCNP|ITIL|CompTIA)\b", re.I)
LANGUAGES = ["English", "Khmer", "French", "Chinese", "Mandarin", "Cantonese", "Japanese", "Korean",
             "Spanish", "German", "Thai", "Vietnamese", "Hindi", "Arabic", "Portuguese", "Russian",
             "Italian", "Indonesian", "Malay", "Lao", "Tagalog", "Dutch", "Bengali", "Urdu"]


def split_sections(lines: list[str], detected_headers: list[str] | None = None) -> dict[str, list[str]]:
    detected = {header_key(h) for h in detected_headers or []}
    sections = {"header": []}
    current = "header"
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        head, _, rest = raw.partition(":")
        key_full, key_head = header_key(raw), header_key(head)
        sec = None
        if len(key_full.split()) <= 5 and (key_full in _ALIAS_TO_SECTION or key_full in detected):
            sec = _ALIAS_TO_SECTION.get(key_full, "other")
            rest = ""
        elif rest.strip() and key_head in _ALIAS_TO_SECTION and len(key_head.split()) <= 3:
            sec = _ALIAS_TO_SECTION[key_head]      # inline "Skills: Python, SQL"
        if sec:
            current = sec
            sections.setdefault(current, [])
            if rest.strip():
                sections[current].append(rest.strip())
        else:
            sections.setdefault(current, []).append(raw)
    return sections


def _parse_date(s: str, is_end: bool, today: date) -> tuple[int, int] | None:
    s = s.strip().lower()
    if s in ("present", "current", "now", "today", "date"):
        return today.year, today.month
    y = re.search(r"(19|20)\d{2}", s)
    if not y:
        return None
    year = int(y.group(0))
    m = re.match(r"([a-z]{3})", s)
    if m and m.group(1) in MONTHS:
        return year, MONTHS[m.group(1)]
    num = re.match(r"(\d{1,2})\s*[/.-]", s)
    if num and 1 <= int(num.group(1)) <= 12:
        return year, int(num.group(1))
    return year, 1


def extract_experience(lines: list[str], today: date | None = None) -> tuple[list[dict], float, str]:
    today = today or date.today()
    entries, intervals = [], []
    for line in lines:
        for m in RANGE_RE.finditer(line):
            start, end = _parse_date(m.group(1), False, today), _parse_date(m.group(2), True, today)
            if not start or not end:
                continue
            s_idx, e_idx = start[0] * 12 + start[1], end[0] * 12 + end[1]
            if not (1960 <= start[0] <= today.year and e_idx >= s_idx and e_idx - s_idx <= 50 * 12):
                continue
            e_idx = min(e_idx, today.year * 12 + today.month)
            intervals.append((s_idx, e_idx))
            entries.append({"text": line[:200], "start": f"{start[0]}-{start[1]:02d}",
                            "end": m.group(2).strip().title() if m.group(2).isalpha() else f"{end[0]}-{end[1]:02d}",
                            "months": e_idx - s_idx})
    months = 0
    for s, e in sorted(_merge(intervals)):
        months += e - s
    computed = round(months / 12, 1)
    explicit = max((float(m.group(1)) for l in lines for m in EXPLICIT_YEARS_RE.finditer(l)), default=0.0)
    if computed >= explicit:
        return entries, computed, "date_ranges" if computed else "not_found"
    return entries, explicit, "explicit_statement"


def _merge(intervals):
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def extract_degrees(lines: list[str]) -> tuple[list[str], int, str | None]:
    found = {}
    for line in lines:
        for level, name, pat in DEGREE_PATTERNS:
            if pat.search(line):
                found[level] = name
                break
    if not found:
        return [], 0, None
    top = max(found)
    return [found[k] for k in sorted(found, reverse=True)], top, found[top]


def degree_level_of(text: str) -> int:
    return extract_degrees([text or ""])[1]


class ProfileBuilder:
    def __init__(self):
        self.skills = get_normalizer()

    def build(self, text: str, core: dict | None = None, today: date | None = None) -> dict:
        core = core or {}
        lines = [l for l in (text or "").splitlines() if l.strip()]
        sections = split_sections(lines, core.get("section_headers"))
        exp_lines = sections.get("experience") or [l for l in lines if l not in sections.get("education", [])]
        edu_lines = sections.get("education") or lines

        skills = self.skills.find_in_text(text)
        entries, years, years_source = extract_experience(exp_lines, today)
        degrees, level, highest = extract_degrees(edu_lines)
        education = [l for l in sections.get("education", []) if len(l.split()) >= 2][:8]
        if not education:
            education = [l for l in lines if any(p.search(l) for _, _, p in DEGREE_PATTERNS)][:5]

        titles, companies = [], []
        for l in exp_lines:
            if len(l.split()) <= 10:
                for m in TITLE_RE.finditer(l):
                    t = re.sub(r"\s+", " ", m.group(1)).strip()
                    if len(t.split()) >= 2 or t.lower() in ("developer", "engineer", "analyst", "accountant"):
                        titles.append(t)
            for m in list(COMPANY_RE.finditer(l)) + list(AT_COMPANY_RE.finditer(l)):
                companies.append(m.group(1).strip(" ,.|"))
        certs = list(sections.get("certifications", []))
        certs += [l for l in lines if CERT_RE.search(l) and len(l.split()) <= 15]
        lang_lines = sections.get("languages", []) + [l for l in lines if l.lower().startswith("language")]
        languages = [lg for lg in LANGUAGES if any(re.search(rf"\b{lg}\b", l, re.I) for l in lang_lines)]

        return {
            "name": core.get("name"), "email": core.get("email"), "phone": core.get("phone"),
            "address": core.get("address"),
            "skills": skills,
            "education": education,
            "degree": degrees,
            "highest_degree": highest,
            "highest_degree_level": level,
            "experience": entries,
            "total_experience_years": years,
            "experience_years_source": years_source,
            "job_titles": list(dict.fromkeys(titles))[:10],
            "companies": list(dict.fromkeys(c for c in companies if len(c) > 2))[:10],
            "certifications": list(dict.fromkeys(certs))[:10],
            "languages": languages,
            "section_headers": core.get("section_headers", []),
            "sections_found": [s for s in sections if s != "header"],
        }
