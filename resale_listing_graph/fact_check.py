"""Deterministic grounding check for a drafted listing.

The seller's confirmed facts are the only source of truth. This check runs in
code (not in an LLM), so it catches the same invented details every time:

- a [SELLER TO ADD] placeholder left in the text
- a dollar sign in the text (the price is set by code, never by the writer)
- risky claim words the seller never said ("handmade", "excellent", "vintage", ...)
- marketing filler the seller never said ("spacious", "charming", "perfect", ...)
- numbers (sizes, measurements) that are not in the seller's facts
- a title that says "unbranded" or "no brand", or is longer than 80 characters
- a mention of "the seller" or "the fact sheet" in the listing text
- a selling point the seller stated ("handmade", "vintage", ...) that the draft leaves out

Judging clarity and buyer questions is left to the LLM critic.
"""

import re

RISKY_WORDS = [
    "handmade", "hand-made", "vintage", "antique", "excellent", "like new", "mint",
    "brand new", "new with tag", "new with tags", "authentic", "genuine", "designer",
    "rare", "smoke-free", "smoke free", "pet-free", "pet free", "sterling", "perfect condition",
    "perfect working order", "never worn", "unworn", "limited edition",
]

FILLER_WORDS = [
    "spacious", "charming", "beautiful", "stunning", "perfect", "playful", "romantic",
    "gorgeous", "must-have", "elegant", "chic", "adorable", "lovely", "versatile",
    "stylish", "trendy", "classic", "timeless", "statement",
]

# Selling points: if the seller's fact sheet says one of these, the draft must say it too.
SELLING_POINTS = [
    "handmade", "hand-made", "vintage", "antique", "sterling", "authentic", "genuine",
    "limited edition", "new with tags", "new with tag", "never worn",
]

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# "not handmade", "isn't vintage", "no sterling": the seller denied it, so do not require it.
NEGATION = r"(?:not|n't|no|never|without)\s+(?:\w+\s+){0,2}"


def _has(word: str, text: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text, re.I) is not None


def _negated(word: str, text: str) -> bool:
    return re.search(NEGATION + re.escape(word), text, re.I) is not None


def check_listing(listing: dict, allowed_text: str, required_text: str = "") -> list[str]:
    """Return problems in `listing` (a dict with title, category, condition, description).

    allowed_text:  everything the seller said; the draft may not go beyond it.
    required_text: the seller's fact sheet; selling points in it must appear in the draft.
    """
    draft = "\n".join(str(listing.get(k, "")) for k in ("title", "category", "condition", "description"))
    allowed = allowed_text.lower()
    issues = []

    if "[SELLER TO ADD" in draft:
        issues.append("A [SELLER TO ADD] placeholder is still in the text. Use only facts in the fact sheet.")
    if "$" in draft:
        issues.append("Do not write a price in the text. The price is shown separately.")

    title = str(listing.get("title", ""))
    if _has("unbranded", title) or _has("no brand", title):
        issues.append('Do not put "unbranded" or "no brand" in the title. Leave the brand out.')
    if len(title) > 80:
        issues.append("The title is longer than 80 characters. Shorten it.")

    if re.search(r"\bsellers?\b|\bfact sheet\b", draft, re.I):
        issues.append('Do not mention the seller or the fact sheet. Write plain facts about the item, for example "No flaws."')

    for word in RISKY_WORDS + FILLER_WORDS:
        if _has(word, draft) and not _has(word, allowed):
            issues.append(f'"{word}" is not in the seller\'s facts. Remove it.')

    for word in SELLING_POINTS:
        if _has(word, required_text) and not _negated(word, required_text) and not _has(word, draft):
            issues.append(f'The seller said "{word}". Include it in the title or description.')

    for number in sorted(set(NUMBER_RE.findall(draft))):
        if number not in allowed:
            issues.append(f"The number {number} is not in the seller's facts. Use only numbers the seller gave.")

    return issues
