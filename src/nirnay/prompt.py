"""Prompt building and reply parsing.

Choices are shown to the model as single-letter codes (A, B, C…) and mapped back in code:
a letter is usually one token, so replies are short and easy to validate.

Choices are either a list of labels, or a mapping of label → description when a label's
meaning isn't obvious from its name (e.g. ``{"cancel": "cancellation fees only"}``).
"""

from __future__ import annotations

import json
import re
import string
from collections.abc import Mapping, Sequence

CODES = string.ascii_uppercase

STRICT_SUFFIX = " Do not write anything else: no words, no punctuation, no explanation."

Choices = Sequence[str] | Mapping[str, str | None]


def label_codes(choices: Choices) -> dict[str, str]:
    """Map ``["billing", "refund"]`` (or a label → description mapping) to
    ``{"A": "billing", "B": "refund"}``."""
    if isinstance(choices, str):
        raise TypeError("choices must be a list of strings, not a single string")
    labels = [str(c).strip() for c in choices]
    if len(labels) < 2:
        raise ValueError("Need at least 2 choices")
    if len(labels) > len(CODES):
        raise ValueError(f"At most {len(CODES)} choices are supported, got {len(labels)}")
    if any(not label for label in labels):
        raise ValueError("Choices must be non-empty strings")
    if len({_norm(label) for label in labels}) != len(labels):
        raise ValueError("Choices must be unique (ignoring case and punctuation)")
    return dict(zip(CODES[: len(labels)], labels, strict=True))


def label_descriptions(choices: Choices) -> dict[str, str]:
    """``{label: description}`` for choices given as a mapping (blank ones skipped);
    empty for a plain list."""
    if not isinstance(choices, Mapping):
        return {}
    return {
        str(label).strip(): str(desc).strip()
        for label, desc in choices.items()
        if desc is not None and str(desc).strip()
    }


def build_messages(
    question: str,
    codes: dict[str, str],
    context: str,
    *,
    descriptions: Mapping[str, str] | None = None,
    strict: bool = False,
) -> list[dict[str, str]]:
    """Static part (question + options + descriptions) goes first so Sarvam's prompt cache
    can reuse it; the per-call context goes last."""
    descriptions = descriptions or {}
    options = "\n".join(
        f"{code} = {label}: {descriptions[label]}" if label in descriptions else f"{code} = {label}"
        for code, label in codes.items()
    )
    letters = ", ".join(codes)
    system = f"{question.strip()}\nOptions:\n{options}\nReply with exactly one letter: {letters}."
    if strict:
        system += STRICT_SUFFIX
    return [{"role": "system", "content": system}, {"role": "user", "content": context}]


# "B", "(B)", "B.", "B) refund", "B: refund", "B = refund", "Answer: B", "Option B"
_LEADING_CODE = re.compile(
    r"^(?:(?:answer|option|label|choice)\s*[:\-]?\s*)?[\(\[]?([A-Za-z])(?:\s*[\)\]\.:=\-]|\s*$)",
    re.IGNORECASE,
)
_PUNCT = " \t\r\n.,:;!?\"'`*()[]{}।"

# How the Latin letters are spelt in Indian scripts. Replying to Hindi, Bengali or Tamil input,
# the model sometimes writes the letter's name ("बी", "বি") instead of the letter ("B").
_SPELT_LETTERS = {
    # Devanagari
    "ए": "A", "बी": "B", "सी": "C", "डी": "D", "ई": "E", "एफ": "F", "जी": "G", "एच": "H",
    "आई": "I", "जे": "J", "के": "K", "एल": "L", "एम": "M", "एन": "N", "ओ": "O", "पी": "P",
    "क्यू": "Q", "आर": "R", "एस": "S", "टी": "T", "यू": "U", "वी": "V", "डब्ल्यू": "W",
    "एक्स": "X", "वाई": "Y", "ज़ेड": "Z", "जेड": "Z",
    # Bengali
    "এ": "A", "বি": "B", "সি": "C", "ডি": "D", "ই": "E", "এফ": "F", "জি": "G", "এইচ": "H",
    "আই": "I", "জে": "J", "কে": "K", "এল": "L", "এম": "M", "এন": "N", "ও": "O", "পি": "P",
    "কিউ": "Q", "আর": "R", "এস": "S", "টি": "T", "ইউ": "U", "ভি": "V", "ডাব্লিউ": "W",
    "ডব্লিউ": "W", "এক্স": "X", "ওয়াই": "Y", "জেড": "Z",
    # Tamil. B/P (பி) and D/T (டி) are spelt the same, so those are left out as ambiguous.
    "ஏ": "A", "சி": "C", "ஈ": "E", "எஃப்": "F", "ஜி": "G", "எச்": "H", "ஹெச்": "H",
    "ஐ": "I", "ஜே": "J", "கே": "K", "எல்": "L", "எம்": "M", "என்": "N", "ஓ": "O",
    "க்யூ": "Q", "ஆர்": "R", "எஸ்": "S", "யூ": "U", "வி": "V", "எக்ஸ்": "X", "வை": "Y",
    "இசட்": "Z",
}  # fmt: skip

# Tamil spells B/P and D/T the same way. These only count when exactly one of the two letters
# is an option (e.g. a positive/negative question has A and B, so "பி" can only mean B).
_AMBIGUOUS_SPELT = {"பி": ("B", "P"), "டி": ("D", "T")}


def parse_reply(text: str | None, codes: dict[str, str]) -> str | None:
    """Return the code the model chose, or ``None`` if the reply doesn't name exactly one.

    Accepts the letter with light decoration (whitespace, brackets, trailing text after a
    separator), a letter alone on the first line followed by an explanation, the letter spelt
    in Devanagari, Bengali or Tamil ("बी", "বি"), the label itself ("refund"), or JSON like
    ``{"label": "B"}``. Anything before a leaked ``</think>`` tag is ignored.
    """
    if text is None:
        return None
    t = text.rsplit("</think>", 1)[-1].strip()
    if not t:
        return None

    if t.startswith("{"):
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and obj:
            value = obj.get("label", next(iter(obj.values())))
            return parse_reply(str(value), codes)

    first_line = t.splitlines()[0].strip()
    for candidate in (t, first_line):
        m = _LEADING_CODE.match(candidate)
        if m and m.group(1).upper() in codes:
            return m.group(1).upper()
        spelt = _SPELT_LETTERS.get(candidate.strip(_PUNCT))
        if spelt in codes:
            return spelt
        options = [c for c in _AMBIGUOUS_SPELT.get(candidate.strip(_PUNCT), ()) if c in codes]
        if len(options) == 1:
            return options[0]

    by_label = {_norm(label): code for code, label in codes.items()}
    return by_label.get(_norm(t))


def _norm(s: str) -> str:
    return " ".join(s.strip(_PUNCT).lower().split())
