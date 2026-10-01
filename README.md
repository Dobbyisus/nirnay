# nirnay

**Sarvam decisions you can trust and budget for.** Pick one label from a fixed list
(route a ticket, detect intent, classify sentiment) using the Sarvam API, and get back the
answer, an honest confidence score, and the exact ₹ cost. When the model is unsure, nirnay
automatically hands the question to Sarvam's thinking mode, so easy decisions stay fast and
cheap and hard ones get the full treatment.

> Status: early development (v0.1.0.dev0). The API may change.

```python
from nirnay import Decider

decider = Decider("sarvam-105b")  # reads SARVAM_API_KEY from the environment

result = decider.decide(
    question="Which team should handle this customer support message?",
    choices={
        "billing": None,
        "refund": "refund policy, getting a refund, or refund status",
        "technical": None,
        "other": None,
    },
    context="Mera paisa do baar kat gaya, refund kab milega?",
)

result.choice          # "refund"
result.confidence      # 1.0  (calibrated if you pass a calibrator; None if escalated)
result.escalated       # False (True when the votes were split and thinking mode decided)
result.votes           # {"refund": 10}
result.usage.cost_inr  # ≈ 0.005
result.latency_ms      # ≈ 200
```

## How it works

1. **Ten votes in one call.** The model answers the same question 10 times in a single request
   (Sarvam's `n` parameter); the prompt is billed once. The most common answer wins, and its
   share of the votes is the confidence.
2. **Escalate when unsure.** If fewer than 8 of 10 votes agree (`escalate_below=0.8`, the
   default), the same question goes to Sarvam's default settings with thinking on, and that
   answer is used. If thinking runs out of tokens without answering, nirnay keeps the vote
   winner instead of failing. Pass `escalate_below=None` to turn escalation off.
3. **Honest confidence.** Fit an `IsotonicCalibrator` on a few hundred labelled examples so that
   "90% confident" means right about 90% of the time, then pass it as `calibrator=`.
   `threshold=` makes non-escalated decisions below that confidence return `choice=None`, for
   a human to handle.

```python
from nirnay import Decider, IsotonicCalibrator

calibrator = IsotonicCalibrator().fit(dev_vote_shares, dev_was_correct)
calibrator.save("calibrator.json")
decider = Decider("sarvam-105b", calibrator=IsotonicCalibrator.load("calibrator.json"))
```

### Describe labels whose names are ambiguous

Pass a mapping instead of a list to tell the model what a label means. In our tests this was
the biggest single accuracy lever: without it, the model was often *confidently* wrong
(e.g. "cancel my order" → `cancel` when `cancel` meant cancellation fees).

```python
choices = {
    "order": "placing, cancelling, changing or tracking an order",
    "cancel": "questions about cancellation FEES or charges only",
    "subscription": "newsletter or promotional email/SMS subscribe or unsubscribe",
    "billing": None,  # obvious from the name; no description needed
}
```

Descriptions go in the cached part of the prompt, so their extra input tokens are mostly
billed at Sarvam's cheaper cached rate.

### Details

- **The fast path runs with thinking off** (`reasoning_effort=None`) and a 4-token answer cap;
  thinking is only used for escalated decisions.
- **Short letter codes.** Choices are shown as A/B/C… and mapped back. Messy replies (`"\nB"`,
  `"b."`, `"refund"`, and letters spelt in Devanagari such as `"बी"`) are understood. If no
  vote is usable, it retries once with a stricter prompt, then escalates.
- **Cache-friendly prompts.** The fixed part (question, options, descriptions) comes first.
- **Retries and pacing.** Handles rate limits (429) and transient errors with backoff.

## Development

Copy `.env.example` to `.env` and add your Sarvam API key.

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```
