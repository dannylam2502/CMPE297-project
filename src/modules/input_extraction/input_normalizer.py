# src/modules/text/normalizer.py
import re

# Common printing ligatures & punctuation, plus a few OCR artifacts
REPLACEMENTS = {
    # ligatures
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    # quotes/dashes/ellipsis
    "“": '"', "”": '"', "„": '"', "‟": '"', "’": "'", "‘": "'",
    "—": "-", "–": "-", "…": "...",
    # stray non-breaking space
    "\u00A0": " ",
}

# Very light heuristics for common OCR spacings/punct
_PUNCT_SPACE = re.compile(r"\s+([,.;:!?])")
_SPACE_PUNCT = re.compile(r"([(\[])\s+")
_PUNCT_OPEN = re.compile(r"\s+([)\]])")

MULTISPACE = re.compile(r"\s+")
WEIRD_WS = re.compile(r"[ \t\r\f\v]+")

def normalize_ocr_asr(text: str) -> str:
    """
    Normalize typical OCR/ASR artifacts while staying conservative.
    - Replace ligatures/publishing punctuation with ASCII equivalents
    - Fix spacing around punctuation/brackets
    - Collapse repeated whitespace
    """
    if not text:
        return text

    # Character-level replacements
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)

    # Collapse weird whitespace
    text = WEIRD_WS.sub(" ", text)

    # Tighten spaces before punctuation: "hello !" -> "hello!"
    text = _PUNCT_SPACE.sub(r"\1", text)
    # Remove space right after opening brackets: "( text" -> "(text"
    text = _SPACE_PUNCT.sub(r"\1", text)
    # Remove space before closing brackets: "text )" -> "text)"
    text = _PUNCT_OPEN.sub(r"\1", text)

    # Collapse multiple spaces and trim
    text = MULTISPACE.sub(" ", text).strip()

    return text