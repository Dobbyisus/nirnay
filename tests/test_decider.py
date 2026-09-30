import pytest

from nirnay import Decider
from nirnay.pricing import cost_inr

CHOICES = ["billing", "refund", "technical", "other"]
Q = "Which team should handle this support message?"
MSG = "Mera paisa do baar kat gaya, refund kab milega?"


def decide(fake, **kwargs):
    with Decider(client=fake.client(), **kwargs) as d:
        return d.decide(Q, CHOICES, MSG)


def test_majority_vote_and_confidence(fake):
    fake.reply(*"BBBBBBBAAA")
    r = decide(fake)
    assert r.choice == r.best_guess == "refund"
    assert r.confidence == pytest.approx(0.7)
    assert r.votes == {"refund": 7, "billing": 3}
    assert r.probabilities == {"refund": 0.7, "billing": 0.3}
    assert r.valid and not r.abstained and not r.retried
    assert r.invalid_samples == 0


def test_request_asks_for_n_samples_with_reasoning_off(fake):
    fake.reply(*"B" * 10)
    decide(fake)
    body = fake.requests[0]
    assert body["n"] == 10
    assert body["temperature"] == 1.0
    assert body["max_tokens"] == 4
    assert "reasoning_effort" in body and body["reasoning_effort"] is None


def test_reasoning_effort_not_sent_to_models_that_reject_it(fake):
    fake.reply(*"B" * 10)
    decide(fake, model="sarvam-105b-conversations")
    assert "reasoning_effort" not in fake.requests[0]


def test_messy_replies_are_normalised(fake):
    fake.reply("\nB", "b.", "refund", "B) refund", "(B)", "B", "B", "B", "A", "C")
    r = decide(fake)
    assert r.votes == {"refund": 8, "billing": 1, "technical": 1}
    assert r.confidence == pytest.approx(0.8)


def test_unusable_replies_lower_confidence(fake):
    fake.reply("B", "B", "B", "B", "B", "B", "hmm", "", None, "maybe")
    r = decide(fake)
    assert r.choice == "refund"
    assert r.confidence == pytest.approx(0.6)
    assert r.invalid_samples == 4


def test_tie_goes_to_first_listed_choice(fake):
    fake.reply(*"DDDDDCCCCC")
    r = decide(fake)
    assert r.best_guess == "technical"
    assert r.confidence == pytest.approx(0.5)


def test_threshold_abstains_when_unsure(fake):
    fake.reply(*"DDDDDCCCCC")
    r = decide(fake, threshold=0.7)
    assert r.abstained and r.choice is None
    assert r.best_guess == "technical" and r.valid


def test_threshold_passes_when_confident(fake):
    fake.reply(*"BBBBBBBBBA")
    r = decide(fake, threshold=0.7)
    assert not r.abstained and r.choice == "refund"


def test_all_unusable_retries_once_with_stricter_prompt(fake):
    fake.reply(*["I think it is a refund issue"] * 10)
    fake.reply(*"B" * 10)
    r = decide(fake)
    assert r.retried and r.valid and r.choice == "refund"
    assert "Do not write anything else" not in fake.requests[0]["messages"][0]["content"]
    assert "Do not write anything else" in fake.requests[1]["messages"][0]["content"]
    assert r.usage.calls == 2


def test_fallback_when_retry_also_fails(fake):
    fake.reply(*["no idea"] * 10)
    fake.reply(*["still no idea"] * 10)
    r = decide(fake, fallback="needs_review")
    assert r.choice == "needs_review" and not r.valid
    assert r.best_guess is None and r.confidence is None and r.votes == {}
    assert r.invalid_samples == 10


def test_single_sample_mode_has_no_confidence(fake):
    fake.reply("B")
    r = decide(fake, samples=1)
    body = fake.requests[0]
    assert "n" not in body and body["temperature"] == 0.0
    assert r.choice == "refund" and r.confidence is None


def test_usage_and_cost_add_up_across_retry(fake):
    fake.reply(*["??"] * 10, prompt_tokens=70, completion_tokens=30)
    fake.reply(*"B" * 10, prompt_tokens=80, cached=64, completion_tokens=20)
    r = decide(fake)
    u = r.usage
    assert (u.input_tokens, u.cached_input_tokens, u.output_tokens) == (150, 64, 50)
    assert u.cost_inr == pytest.approx((86 * 29.28 + 64 * 10.98 + 50 * 73.20) / 1e6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": 0},
        {"samples": 129},
        {"temperature": 3},
        {"threshold": 1.5},
        {"samples": 1, "threshold": 0.5},
    ],
)
def test_invalid_settings_rejected(fake, kwargs):
    with pytest.raises(ValueError):
        Decider(client=fake.client(), **kwargs)


def test_unknown_model_has_no_cost():
    assert cost_inr("some-new-model", 100, 0, 10) is None
