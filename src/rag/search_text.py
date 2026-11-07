"""Versioned normalization for derived lexical text; source text stays intact."""

import re
import unicodedata

NORMALIZATION_VERSION = "lexical-1"
_TRANSLATION = str.maketrans(
    "يك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹\u200c\u200d",
    "یک01234567890123456789  ",
)


def normalize_search_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().translate(_TRANSLATION).replace("ـ", "")
    # Arabic vowel marks are optional in Persian/Arabic text.
    text = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    return " ".join(text.split())
