import re

# Simple slang and alias normalization
# This list can grow over time based on real user questions
ALIASES = {
    r"\bobamacare\b": "affordable care act aca",
    r"\bfood stamps\b": "snap",
    r"\bwelfare\b": "tanf snap medicaid",
    r"\bmedicare for all\b": "single payer medicare expansion",
    r"\bdebt relief\b": "student loan forgiveness",
    r"\btiktok ban\b": "tiktok divest ban",
    r"\bborder money\b": "border security funding",
    r"\bukraine money\b": "ukraine aid funding",
    r"\bgun control\b": "firearms background checks",
    r"\bde(i|&i)\b": "diversity equity inclusion",
}

# Remove punctuation but keep words and numbers
CLEAN_RE = re.compile(r"[^a-z0-9\s\-]")

def normalize_query(query: str) -> str:
    """
    Normalize slang, aliases, and punctuation
    so search works even with casual language.
    """
    text = query.lower().strip()

    for pattern, replacement in ALIASES.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    text = CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text
