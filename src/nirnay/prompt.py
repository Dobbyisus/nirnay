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

# How the Latin letters are spelt in Devanagari. Replying to Hindi input, the model sometimes
# writes the letter's name ("बी") instead of the letter ("B").
_DEVANAGARI_LETTERS = {
    "ए": "A", "बी": "B", "सी": "C", "डी": "D", "ई": "E", "एफ": "F", "जी": "G", "एच": "H",
    "आई": "I", "जे": "J", "के": "K", "एल": "L", "एम": "M", "एन": "N", "ओ": "O", "पी": "P",
    "क्यू": "Q", "आर": "R", "एस": "S", "टी": "T", "यू": "U", "वी": "V", "डब्ल्यू": "W",
    "एक्स": "X", "वाई": "Y", "ज़ेड": "Z", "जेड": "Z",
}  # fmt: skip


def parse_reply(text: str | None, codes: dict[str, str]) -> str | None:
    """Return the code the model chose, or ``None`` if the reply doesn't name exactly one.

    Accepts the letter with light decoration (whitespace, brackets, trailing text after a
    separator), the letter spelt in Devanagari ("बी"), the label itself ("refund"), or JSON
    like ``{"label": "B"}``.
    """
    if text is None:
        return None
    t = text.strip()
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

    m = _LEADING_CODE.match(t)
    if m and m.group(1).upper() in codes:
        return m.group(1).upper()

    spelt = _DEVANAGARI_LETTERS.get(t.strip(_PUNCT))
    if spelt in codes:
        return spelt

    by_label = {_norm(label): code for code, label in codes.items()}
    return by_label.get(_norm(t))


def _norm(s: str) -> str:
    return " ".join(s.strip(_PUNCT).lower().split())
