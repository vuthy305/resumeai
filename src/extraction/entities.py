"""BIO -> entity reconstruction shared by the baseline and LayoutLMv3."""
from src.utils.text_utils import find_emails, find_phones, normalize_whitespace


def bio_to_entities(words: list[str], labels: list[str], page: int = 1) -> list[dict]:
    entities, current = [], None
    for i, (word, label) in enumerate(zip(words, labels)):
        if not label or label == "O":
            current = None
            continue
        prefix, etype = label.split("-", 1) if "-" in label else ("B", label)
        if prefix == "B" or current is None or current["entity"] != etype:
            current = {"entity": etype, "tokens": [word], "word_indices": [i], "page": page}
            entities.append(current)
        else:
            current["tokens"].append(word)
            current["word_indices"].append(i)
    for e in entities:
        e["text"] = normalize_whitespace(" ".join(e.pop("tokens")))
    return entities


def _clean_label_prefix(text: str) -> str:
    for p in ("address", "addr", "name", "location"):
        if text.lower().startswith(p + ":"):
            return text[len(p) + 1:].strip()
    return text.strip(" |,;:•")


def entities_to_fields(entities: list[dict]) -> dict:
    fields = {"name": None, "email": None, "phone": None, "address": None,
              "section_headers": [], "evidence": []}
    for e in entities:
        fields["evidence"].append({"text": e["text"], "entity": e["entity"], "page": e["page"]})
        t = e["entity"]
        if t == "NAME" and not fields["name"]:
            fields["name"] = _clean_label_prefix(e["text"]) or None
        elif t == "EMAIL" and not fields["email"]:
            emails = find_emails(e["text"])
            fields["email"] = emails[0] if emails else None
        elif t == "PHONE" and not fields["phone"]:
            phones = find_phones(e["text"])
            fields["phone"] = phones[0] if phones else None
        elif t == "ADDRESS" and not fields["address"]:
            fields["address"] = _clean_label_prefix(e["text"]) or None
        elif t == "SECTIONHEADER":
            h = e["text"].strip(" :")
            if h and h not in fields["section_headers"]:
                fields["section_headers"].append(h)
    return fields


def merge_page_fields(pages: list[dict]) -> dict:
    """Merge per-page field dicts: first non-empty scalar wins, lists are concatenated."""
    merged = {"name": None, "email": None, "phone": None, "address": None,
              "section_headers": [], "evidence": []}
    for f in pages:
        for k in ("name", "email", "phone", "address"):
            merged[k] = merged[k] or f.get(k)
        for h in f.get("section_headers", []):
            if h not in merged["section_headers"]:
                merged["section_headers"].append(h)
        merged["evidence"].extend(f.get("evidence", []))
    return merged
