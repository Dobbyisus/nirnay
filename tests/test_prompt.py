import pytest

from nirnay.prompt import build_messages, label_codes, label_descriptions, parse_reply

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


def test_label_codes_accepts_description_mapping():
    choices = {"order": "placing or cancelling orders", "cancel": "cancellation fees only"}
    assert label_codes(choices) == {"A": "order", "B": "cancel"}


def test_label_descriptions_skip_blanks_and_plain_lists():
    assert label_descriptions(["order", "cancel"]) == {}
    assert label_descriptions({"order": "  orders ", "cancel": None, "refund": " "}) == {
        "order": "orders"
    }


def test_build_messages_includes_descriptions_where_given():
    codes = {"A": "order", "B": "cancel"}
    system, _ = build_messages(
        "Which team?", codes, "x", descriptions={"cancel": "cancellation fees only"}
    )
    assert "A = order\nB = cancel: cancellation fees only\n" in system["content"]


def test_ambiguous_tamil_letters_resolve_only_when_one_option_fits():
    two = {"A": "positive", "B": "negative"}
    assert parse_reply("பி", two) == "B"  # P isn't an option, so it can only be B
    assert parse_reply("டி", two) is None  # neither D nor T is an option
    many = {c: c.lower() for c in "ABCDEFGHIJKLMNOPQR"}
    assert parse_reply("பி", many) is None  # B and P are both options: ambiguous
    assert parse_reply("டி", many) == "D"  # T isn't an option


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
        ("बी", "B"),
        ("डी।", "D"),
        (" सी ", "C"),
        ("ए", "A"),
        ("বি", "B"),  # Bengali
        ("এ", "A"),
        ("সি।", "C"),
        ("சி", "C"),  # Tamil
        ("I</think>\nC", "C"),  # leaked reasoning tag
        ("D\n\n(This request is about something else)", "D"),  # letter, then explanation
    ],
)
def test_parse_reply_accepts_decorated_answers(reply, expected):
    assert parse_reply(reply, CODES) == expected


@pytest.mark.parametrize(
    "reply",
    [
        None,
        "",
        "   ",
        "E",
        "Billing or refund",
        "A refund request",
        "I think",
        "{not json",
        "ई",  # Devanagari "E": not one of the four codes
        "मैं",  # a word, not a letter name
        "Answer:\nmaybe",
    ],
)
def test_parse_reply_rejects_unusable_answers(reply):
    assert parse_reply(reply, CODES) is None
