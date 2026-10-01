import pytest

from nirnay import Decider
from nirnay.pricing import cost_inr

from .conftest import completion

CHOICES = ["billing", "refund", "technical", "other"]
Q = "Which team should handle this support message?"
MSG = "Mera paisa do baar kat gaya, refund kab milega?"


def decide(fake, **kwargs):
    """Voting path only, unless a test turns escalation on explicitly."""
    kwargs.setdefault("escalate_below", None)
    with Decider(client=fake.client(), **kwargs) as d:
        return d.decide(Q, CHOICES, MSG)


def thinking_reply(content, *, prompt_tokens=70, completion_tokens=600, finish="stop"):
    body = completion([content], prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    body["choices"][0]["finish_reason"] = finish
    return body


def test_escalation_is_on_by_default_for_thinking_models(fake):
    with Decider(client=fake.client()) as d:
        assert d.escalate_below == 0.8
    with Decider("sarvam-105b-conversations", client=fake.client()) as d:
        assert d.escalate_below is None


def test_confident_votes_are_not_escalated(fake):
    fake.reply(*"BBBBBBBBBA")
    r = decide(fake, escalate_below=0.8)
    assert not r.escalated and r.choice == "refund" and r.confidence == pytest.approx(0.9)
    assert len(fake.requests) == 1


def test_split_votes_escalate_to_thinking_with_the_same_prompt(fake):
    fake.reply(*"BBBBBBAAAA")
    fake.push(200, thinking_reply("A"))
    r = decide(fake, escalate_below=0.8)
    vote_req, think_req = fake.requests
    assert think_req["messages"] == vote_req["messages"]
    # Sarvam defaults: no reasoning_effort (thinking on), one answer, server max_tokens.
    assert set(think_req) == {"model", "messages"}
    assert r.escalated and r.escalation_answer == "billing"
    assert r.choice == r.best_guess == "billing"
    assert r.confidence is None and r.raw_confidence == pytest.approx(0.6)
    assert r.votes == {"refund": 6, "billing": 4} and r.valid and not r.abstained


def test_escalation_counts_both_calls_in_usage(fake):
    fake.reply(*"BBBBBAAAAA", prompt_tokens=70, cached=64, completion_tokens=30)
    fake.push(200, thinking_reply("B", prompt_tokens=70, completion_tokens=600))
    r = decide(fake, escalate_below=0.8)
    u = r.usage
    assert (u.calls, u.input_tokens, u.cached_input_tokens, u.output_tokens) == (2, 140, 64, 630)


def test_thinking_without_an_answer_keeps_the_vote_winner(fake):
    fake.reply(*"BBBBBBAAAA")
    fake.push(200, thinking_reply("", finish="length"))  # thought until max_tokens ran out
    r = decide(fake, escalate_below=0.8)
    assert r.escalated and r.escalation_answer is None
    assert r.choice == "refund" and r.valid


def test_thinking_answer_in_devanagari_is_understood(fake):
    fake.reply(*"BBBBBBAAAA")
    fake.push(200, thinking_reply("डी"))
    r = decide(fake, escalate_below=0.8)
    assert r.escalation_answer == "other"


def test_unusable_votes_escalate_instead_of_falling_back(fake):
    fake.reply(*["??"] * 10)
    fake.reply(*["still ??"] * 10)
    fake.push(200, thinking_reply("C"))
    r = decide(fake, escalate_below=0.8, fallback="needs_review")
    assert r.retried and r.escalated and r.choice == "technical" and r.valid


def test_escalated_decisions_never_abstain(fake):
    fake.reply(*"BBBBBAAAAA")
    fake.push(200, thinking_reply("A"))
    r = decide(fake, escalate_below=0.8, threshold=0.9)
    assert not r.abstained and r.choice == "billing"


def test_escalation_max_tokens_is_passed_when_set(fake):
    fake.reply(*"BBBBBAAAAA")
    fake.push(200, thinking_reply("A"))
    decide(fake, escalate_below=0.8, escalation_max_tokens=4096)
    assert fake.requests[1]["max_tokens"] == 4096


@pytest.mark.parametrize(
    "kwargs",
    [
        {"escalate_below": 0.0},
        {"escalate_below": 1.5},
        {"escalate_below": 0.8, "samples": 1},
        {"escalate_below": 0.8, "model": "sarvam-105b-conversations"},
    ],
)
def test_invalid_escalation_settings_rejected(fake, kwargs):
    with pytest.raises(ValueError):
        Decider(client=fake.client(), **kwargs)


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


def test_described_choices_reach_the_prompt_and_return_label_names(fake):
    fake.reply(*"AAAAAAAAAB")
    choices = {
        "order": "placing, cancelling, changing or tracking an order",
        "cancel": "questions about cancellation FEES only",
    }
    with Decider(client=fake.client()) as d:
        r = d.decide(Q, choices, "bhaiya cancel kro order")
    system = fake.requests[0]["messages"][0]["content"]
    assert "A = order: placing, cancelling, changing or tracking an order" in system
    assert "B = cancel: questions about cancellation FEES only" in system
    assert r.choice == "order" and r.votes == {"order": 9, "cancel": 1}


def test_descriptions_kept_in_strict_retry(fake):
    fake.reply(*["hmm"] * 10)
    fake.reply(*"B" * 10)
    with Decider(client=fake.client()) as d:
        d.decide(Q, {"order": "orders", "cancel": "fees only"}, "x")
    assert "B = cancel: fees only" in fake.requests[1]["messages"][0]["content"]


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
