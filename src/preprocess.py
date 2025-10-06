from __future__ import annotations
import re
from typing import Iterable, List, Optional

try:
    from hazm import Normalizer
    _hz = Normalizer()
except Exception:
    _hz = None

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_TAG_RE = re.compile(r"(@[\w_]+|#\w+)")
SPACE_RE = re.compile(r"\s+")

def normalize_fa(text: str) -> str:
    if not text:
        return ""
    if _hz:
        text = _hz.normalize(text)
    # unify Arabic chars to Persian forms (basic)
    text = text.replace("ي", "ی").replace("ك", "ک")
    return text

def clean_text(text: str, lang: str = "fa") -> str:
    text = text or ""
    if lang == "fa":
        text = normalize_fa(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text

def batch_clean(texts: Iterable[str], lang: str = "fa") -> List[str]:
    return [clean_text(t, lang=lang) for t in texts]