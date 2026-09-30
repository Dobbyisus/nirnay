# nirnay

**Sarvam decisions you can trust and budget for.** Pick one label from a fixed list
(route a ticket, detect intent, classify sentiment) using the Sarvam API, and get back a
confidence score, the option to escalate when the model is unsure, and the exact ₹ cost.

> Status: early development (v0.1.0.dev0). The API may change. The benchmark is not done yet.

```python
from nirnay import Decider

decider = Decider("sarvam-105b", threshold=0.7)  # reads SARVAM_API_KEY from the environment

result = decider.decide(
    question="Which team should handle this customer support message?",
    choices=["billing", "refund", "technical", "other"],
    context="Mera paisa do baar kat gaya, refund kab milega?",
)

result.choice          # "refund"  (None if confidence < threshold → escalate to a human)
result.confidence      # 1.0       (share of the 10 samples that agreed)
result.votes           # {"refund": 10}
result.usage.cost_inr  # ≈ 0.003
result.latency_ms      # ≈ 270
```

## How it works

- **One call, ten votes.** The model answers the same question 10 times in a single request
  (Sarvam's `n` parameter). You pay for the prompt once. The most common answer wins, and
  its share of the votes is the confidence.
- **Reasoning off by default.** For a one-word answer, hidden "thinking" mostly adds cost and
  delay. In our tests, turning it off was ~9× faster and ~5× cheaper, with the same answer.
- **Short letter codes.** Choices are shown as A/B/C… and mapped back, so answers are one
  token and easy to validate. Messy replies (`"\nB"`, `"b."`, `"refund"`) are normalised; if
  none are usable, it retries once with a stricter prompt, then returns your `fallback`.
- **Cache-friendly prompts.** The fixed part (question + options) comes first, so Sarvam's
  prompt cache can bill it at the cheaper cached rate.
- **Retries and pacing.** Handles rate limits (429) and transient errors with backoff.

Raw vote shares are not yet calibrated: "9 out of 10" doesn't guarantee 90% accuracy.
Calibration and a benchmark on English, Hindi and Hinglish are next.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```
