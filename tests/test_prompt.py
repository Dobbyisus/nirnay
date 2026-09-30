import pytest

from nirnay.prompt import build_messages, label_codes, parse_reply

CODES = {"A": "billing", "B": "refund", "C": "technical", "D": "other"}


def test_label_codes_maps_letters_in_order():
    assert label_codes(["billing", "refund"]) == {"A": "billing", "B": "refund"}


@pytest.mark.parametrize(
    "choices, error",
    [
        (["only"], ValueError),
        (["a", " "], ValueError),
        (["Refund", "refund."], ValueError),
        ([str(i) for i in range(27)], ValueError),
        ("billing", TypeError),
    ],
)
def test_label_codes_rejects_bad_choices(choices, error):
    with pytest.raises(error):
        label_codes(choices)


def test_build_messages_puts_static_part_first():
    system, user = build_messages("Which team?", CODES, "paisa kat gaya")
    assert system["role"] == "system" and user == {"role": "user", "content": "paisa kat gaya"}
    assert "A = billing\nB = refund" in system["content"]
    assert system["content"].endswith("Reply with exactly one letter: A, B, C, D.")


def test_strict_prompt_adds_instruction():
    system, _ = build_messages("Which team?", CODES, "x", strict=True)
    assert "Do not write anything else" in system["content"]


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("B", "B"),
        ("\nB", "B"),
        ("b.", "B"),
        ("(C)", "C"),
        ("A) billing", "A"),
        ("D: other", "D"),
        ("B = refund", "B"),
        ("Answer: C", "C"),
        ("Option D", "D"),
        ("refund", "B"),
        ("  Technical. ", "C"),
        ('{"label": "A"}', "A"),
        ('{"label": "refund"}', "B"),
    ],
)
def test_parse_reply_accepts_decorated_answers(reply, expected):
    assert parse_reply(reply, CODES) == expected


@pytest.mark.parametrize(
    "reply",
    [None, "", "   ", "E", "Billing or refund", "A refund request", "I think", "{not json"],
)
def test_parse_reply_rejects_unusable_answers(reply):
    assert parse_reply(reply, CODES) is None
