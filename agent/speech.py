"""Turn written text into something a TTS voice reads the way a person would.

Only the audio is changed; the on-screen transcript keeps the written form.

    "+91 97734 43748"        -> "plus 9 1, 9 7 7 3 4, 4 3 7 4 8"
    "UOWI-Admit@uow.edu.au"  -> "U O W I dash admit at U O W dot E D U dot A U"
    "uow.edu.au/india"       -> "U O W dot E D U dot A U slash india"
    "AUD 27,900"             -> "27,900 Australian dollars"
    "50%"                    -> "50 percent"

plus the profile's lexicon (e.g. "IELTS => eye elts").
"""

from __future__ import annotations

import re

_TLDS = r"(?:com|in|au|org|net|edu|ac|gov|io|co|ae|uk|ai)"
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(
    rf"\b(?:https?://)?(?:www\.)?[a-zA-Z][\w-]*(?:\.[a-zA-Z][\w-]*)*\.{_TLDS}\b(?:/[\w\-/]*)?",
    re.I,
)
# 8+ digits, optionally with a leading + and spaces/dashes between groups.
_PHONE = re.compile(r"(?<![\w,.])\+?\d[\d \-]{6,}\d(?![\w,])")
_PIN = re.compile(r"\b(PIN(?: code)?\s+)(\d{6})\b", re.I)
_AMOUNT = r"(\d[\d,]*(?:\.\d+)?(?:\s?(?:lakhs?|crores?|thousand|million))?)"
_CURRENCY = [
    (re.compile(rf"\bAUD\s?\$?\s?{_AMOUNT}"), r"\1 Australian dollars"),
    (re.compile(rf"\b(?:INR|Rs\.?)\s?{_AMOUNT}"), r"\1 rupees"),
    (re.compile(rf"₹\s?{_AMOUNT}"), r"\1 rupees"),
    (re.compile(rf"(?<![\w])\${_AMOUNT}"), r"\1 dollars"),
]
_SEPARATORS = {"@": " at ", ".": " dot ", "-": " dash ", "_": " underscore ", "/": " slash ", "+": " plus "}


def _spell_token(tok: str) -> str:
    """Short or non-word tokens are spelled out letter by letter."""
    if tok.isdigit():
        return " ".join(tok)
    if len(tok) <= 4 or not re.search(r"[aeiou]", tok, re.I):
        return " ".join(tok.upper())
    return tok.lower()


def _spell_address(addr: str) -> str:
    addr = re.sub(r"^https?://", "", addr)
    parts = re.split(r"([@.\-_/+])", addr.rstrip("/"))
    out = []
    for p in parts:
        if p in _SEPARATORS:
            out.append(_SEPARATORS[p])
        elif p:
            out.append(_spell_token(p))
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _spell_phone(match: re.Match) -> str:
    raw = match.group(0)
    if sum(c.isdigit() for c in raw) < 8:
        return raw
    groups = [g for g in re.split(r"[ \-]+", raw) if g]
    spoken = []
    for g in groups:
        prefix = "plus " if g.startswith("+") else ""
        spoken.append(prefix + " ".join(g.lstrip("+")))
    return ", ".join(spoken)


def _apply_lexicon(text: str, lexicon: dict[str, str]) -> str:
    if not lexicon:
        return text
    # Longest phrases first, so "UOW India" wins over "UOW".
    keys = sorted(lexicon, key=len, reverse=True)
    pattern = re.compile(r"(?<![\w])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w])")
    return pattern.sub(lambda m: lexicon[m.group(1)], text)


_INDIAN_GROUPED = re.compile(r"\b\d{1,2}(?:,\d{2})+,\d{3}\b")


def _say_indian_number(match: re.Match) -> str:
    """1,50,000 -> "1 lakh 50 thousand"; 1,25,00,000 -> "1 crore 25 lakh"."""
    n = int(match.group(0).replace(",", ""))
    parts = []
    for size, name in ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand")):
        if n >= size:
            parts.append(f"{n // size} {name}")
            n %= size
    if n:
        parts.append(str(n))
    return " ".join(parts)


def speakable(text: str, lexicon: dict[str, str] | None = None) -> str:
    text = _EMAIL.sub(lambda m: _spell_address(m.group(0)), text)
    text = _URL.sub(lambda m: _spell_address(m.group(0)), text)
    text = _PIN.sub(lambda m: m.group(1) + " ".join(m.group(2)), text)
    for pattern, repl in _CURRENCY:  # before lakh conversion: "Rs. 1,50,000" -> "1,50,000 rupees"
        text = pattern.sub(repl, text)
    text = _INDIAN_GROUPED.sub(_say_indian_number, text)
    text = _PHONE.sub(_spell_phone, text)
    text = re.sub(r"(\d)\s?%", r"\1 percent", text)
    text = _apply_lexicon(text, lexicon or {})
    return re.sub(r"\s+", " ", text).strip()
