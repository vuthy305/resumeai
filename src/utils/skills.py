"""Skill normalisation driven by data/skill_dictionary.json."""
import json
import re
from functools import lru_cache

from src.config import SKILL_DICT_PATH


def _alias_pattern(alias: str) -> str:
    return r"\s+".join(re.escape(part) for part in alias.split())


class SkillNormalizer:
    def __init__(self, path=SKILL_DICT_PATH):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.entries = data["skills"]
        self.alias_map, self.patterns = {}, []
        for canonical, meta in self.entries.items():
            names = set(meta.get("aliases", []))
            if meta.get("match_canonical", True):
                names.add(canonical)
            aliases = sorted(names, key=len, reverse=True)
            self.alias_map[canonical.lower()] = canonical
            for a in aliases:
                self.alias_map[a.lower()] = canonical
            body = "|".join(_alias_pattern(a) for a in aliases)
            flags = 0 if meta.get("case_sensitive") else re.IGNORECASE
            self.patterns.append((canonical, re.compile(rf"(?<![A-Za-z0-9+#])(?:{body})(?![A-Za-z0-9+#])", flags)))

    def normalize(self, skill: str) -> str:
        s = re.sub(r"\s+", " ", (skill or "").strip())
        return self.alias_map.get(s.lower(), s)

    def normalize_list(self, skills) -> list[str]:
        return list(dict.fromkeys(self.normalize(s) for s in skills or [] if s and s.strip()))

    def find_in_text(self, text: str) -> list[str]:
        hits = []
        for canonical, pat in self.patterns:
            m = pat.search(text or "")
            if m:
                hits.append((m.start(), canonical))
        return [c for _, c in sorted(hits)]


@lru_cache(maxsize=1)
def get_normalizer() -> SkillNormalizer:
    return SkillNormalizer()
