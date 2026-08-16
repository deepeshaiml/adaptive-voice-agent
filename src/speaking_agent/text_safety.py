from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    canonical = unicodedata.normalize("NFKC", text).translate(
        str.maketrans({"’": "'", "‘": "'", "`": "'"})
    )
    return " ".join(canonical.casefold().split())


def normalize_match_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w']+", " ", normalize_text(text)).split())


def claims_human_identity(text: str) -> bool:
    normalized = normalize_text(text)
    identity = (
        r"(?:(?:an?|real|actual|actually|live)\s+){0,3}"
        r"(?:human|person)(?:\s+(?:caller|agent))?"
    )
    return re.search(
        rf"\b(?:i am|i'm|im)\s+{identity}\b|"
        rf"\b(?:i am|i'm|im)\s+"
        rf"(?:(?:an?|real|actual|actually)\s+){{1,2}}agent\b|"
        rf"\b(?:this is|you(?:'re| are)\s+(?:speaking|talking)\s+(?:to|with))\s+"
        rf"{identity}\b",
        normalized,
    ) is not None
