# nirnay

**Fast, cheap decisions with honest confidence on the Sarvam API.**

nirnay (Hindi: *decision*) is a small Python library for the questions inside your product that
have a fixed set of answers: which team should handle this ticket, what does this user want, is
this review positive. It returns the answer, a confidence score you can act on, and the exact ₹
cost, and it hands genuinely hard cases to Sarvam's thinking mode automatically.

Benchmarked on 1,616 test decisions across 5 tasks in English, Hindi, Hinglish, Bengali and Tamil,
against `sarvam-105b` with Sarvam's default settings:

| | Sarvam default settings | nirnay |
|---|---|---|
| Accuracy (with label descriptions) | 89.7% | **91.2%** |
| Cost per 1,000 decisions | ₹43.51 | **₹11.97** (28%) |
| Typical response time | 2.2–3.5 s | **0.17–0.19 s** (12–21× faster) |
| Decisions with no usable answer | 1.2–5.4% | **0%** |
| Confidence score | — | **yes** |

> **Fine print:** most of the accuracy gap comes from the default sometimes giving no answer
> (it thinks until it hits the token limit). On the decisions it did answer, the default was
> slightly more accurate on 4 of 5 tasks. Read it as **similar accuracy, much lower cost and
> latency, and no failed decisions.** Full results and caveats are [below](#benchmark).

![Accuracy by task](https://raw.githubusercontent.com/Dobbyisus/nirnay/main/assets/benchmarks/01_accuracy_with_descriptions.png)

---

## Install

```bash
pip install nirnay
```

Requires Python 3.10+ and a Sarvam API key ([dashboard.sarvam.ai](https://dashboard.sarvam.ai)),
read from the `SARVAM_API_KEY` environment variable.

## Quickstart

```python
from nirnay import Decider

decider = Decider("sarvam-105b")

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
result.confidence      # 1.0 (share of votes; calibrated if you pass a calibrator)
result.escalated       # False (True when votes were split and thinking mode decided)
result.votes           # {"refund": 10}
result.usage.cost_inr  # ≈ 0.005
result.latency_ms      # ≈ 180
```

## How it works

1. **Ten votes in one call.** The model answers the same question 10 times in a single request
   (Sarvam's `n` parameter); the prompt is billed once. The most common answer wins, and the
   share of votes it got is the confidence.
2. **Escalate when unsure.** If fewer than 8 of 10 votes agree, nirnay asks again with Sarvam's
   default settings (thinking on) and uses that answer. Across our benchmark this happened for
   2–20% of decisions. If thinking runs out of tokens without answering, nirnay keeps the vote
   winner instead of failing.
3. **Honest confidence.** Optionally fit a calibrator on a few hundred labelled examples so that
   "90% confident" means right about 90% of the time. Set `threshold=` to get `choice=None`
   (route to a human) when confidence is too low.
4. **Label descriptions.** Pass `{label: description}` instead of a list to say what a category
   means. This was the biggest single accuracy lever in our tests (+17 points on support routing).
5. **Robust parsing.** Replies such as `"\nB"`, `"b."`, `"refund"`, or a letter spelt in Hindi,
   Bengali or Tamil script (`"बी"`, `"বি"`, `"சி"`) are all understood.

### Details

- The fast voting path runs with thinking off (`reasoning_effort=None`) and a 4-token answer cap;
  thinking is used only for escalated decisions.
- Choices are shown to the model as letters (A, B, C…) and mapped back, so answers are one token.
- The fixed part of the prompt (question, options, descriptions) comes first so Sarvam's prompt
  cache can bill it at the cached rate.
- Rate limits (429) and transient errors are retried with backoff; `min_interval_s` paces calls.

![Label descriptions](https://raw.githubusercontent.com/Dobbyisus/nirnay/main/assets/benchmarks/08_label_descriptions.png)

## Calibration

```python
from nirnay import Decider, IsotonicCalibrator

# raw vote shares and whether the vote winner was right, from a labelled dev set
calibrator = IsotonicCalibrator().fit(dev_vote_shares, dev_was_correct)
calibrator.save("calibrator.json")

decider = Decider(
    "sarvam-105b",
    calibrator=IsotonicCalibrator.load("calibrator.json"),
    threshold=0.7,  # below this, choice is None: send it to a human
)
```

Calibrate when you have a few hundred labelled examples and the raw vote share looks
overconfident. With small or noisy dev sets the raw vote share is often already a reasonable
confidence (see the benchmark).

![Confidence honesty](https://raw.githubusercontent.com/Dobbyisus/nirnay/main/assets/benchmarks/11_confidence_honesty.png)

## API

**`Decider(model="sarvam-105b", *, samples=10, temperature=None, escalate_below=AUTO,
escalation_max_tokens=None, threshold=None, calibrator=None, fallback=None,
reasoning_effort=None, max_tokens=4, client=None, endpoint="/v1/chat/completions")`**

- `samples`: votes per decision (1–128). `1` turns voting off (no confidence).
- `escalate_below`: escalate when the winner's vote share is below this. Defaults to 0.8 for
  models with thinking; `None` turns escalation off.
- `threshold`: non-escalated decisions with confidence below this return `choice=None`.
- `calibrator`: any `float -> float` callable, e.g. a fitted `IsotonicCalibrator`.
- `fallback`: returned as `choice` if nothing usable came back at all.

**`decider.decide(question, choices, context) -> Decision`**: `choices` is a list of labels, or a
mapping `{label: description or None}`.

**`Decision`** fields: `choice`, `best_guess`, `confidence`, `raw_confidence`, `votes`,
`probabilities`, `escalated`, `escalation_answer`, `abstained`, `valid`, `invalid_samples`,
`retried`, `usage` (`input_tokens`, `cached_input_tokens`, `output_tokens`, `cost_inr`, `calls`),
`latency_ms`, `model`.

**Calibration helpers:** `IsotonicCalibrator` (`fit`, `save`, `load`), `expected_calibration_error`,
and in `nirnay.calibration`: `reliability_bins`, `risk_coverage`.

**`SarvamClient(api_key=None, *, timeout=30, max_retries=3, min_interval_s=0)`** if you need to
share a client or pace requests.

## Benchmark

Two systems, same model (`sarvam-105b`), same prompt, same categories and descriptions, same
answer parser:

- **Sarvam default settings:** only the model and messages are sent (thinking on, one answer).
- **nirnay:** the library as shipped.

| Task | Data | Languages | Categories | Test n |
|---|---|---|---|---|
| Support routing | 352 realistic support messages (AI-drafted, see limitations) | Hindi, Hinglish | 11 | 242 |
| Voice-assistant domain | Google Hinglish-TOP (same request in two forms) | English, Hinglish | 8 | 240 |
| Voice-assistant scenario | Amazon MASSIVE (same request in four languages) | English, Hindi, Bengali, Tamil | 18 | 504 |
| Review sentiment | AI4Bharat IndicSentiment (parallel reviews) | English, Hindi, Bengali, Tamil | 2 | 480 |
| Tweet sentiment | SemEval-2020 SentiMix (real code-mixed tweets) | Hinglish | 3 | 150 |

### Results with label descriptions

| Task | Default | nirnay | Diff (95% CI) | Default failed | ₹/1k default → nirnay | p50 | Escalated |
|---|---|---|---|---|---|---|---|
| Support routing | 90.5% | 90.5% | 0.0 (−3.3, +3.3) | 3.3% | 42.90 → 12.71 | 2.19 → 0.18 s | 11.2% |
| Voice domain | 95.8% | 96.2% | +0.4 (0.0, +1.2) | 1.2% | 35.20 → 7.39 | 2.50 → 0.19 s | 3.8% |
| Voice scenario | 86.1% | 88.9% | **+2.8** (+0.8, +5.0) | 5.4% | 47.84 → 18.38 | 3.21 → 0.19 s | 16.5% |
| Review sentiment | 97.5% | 99.2% | **+1.7** (+0.6, +2.9) | 1.5% | 39.45 → 5.35 | 2.86 → 0.18 s | 2.3% |
| Tweet sentiment | 65.3% | 66.7% | +1.3 (−2.7, +5.3) | 5.3% | 56.22 → 17.75 | 3.55 → 0.17 s | 20.0% |

Bold = statistically significant (paired, exact McNemar p < 0.05). Without label descriptions the
two systems are within noise of each other (support 75.2% vs 73.1%, voice scenario 76.2% vs
78.2%, tweets 66.7% vs 64.7%), and nirnay escalates more and saves less (costs 12–54% of the
default).

![Cost per 1,000 decisions](https://raw.githubusercontent.com/Dobbyisus/nirnay/main/assets/benchmarks/03_cost_per_1000.png)
![Typical response time](https://raw.githubusercontent.com/Dobbyisus/nirnay/main/assets/benchmarks/04_latency_typical.png)
![Failed answers](https://raw.githubusercontent.com/Dobbyisus/nirnay/main/assets/benchmarks/06_failed_answers.png)
![Default accuracy when it answered](https://raw.githubusercontent.com/Dobbyisus/nirnay/main/assets/benchmarks/07_default_answered_only.png)

### Caveats

- **Failed answers.** The default's unanswered decisions count as wrong. On the decisions it did
  answer it scored 93.6 / 97.0 / 91.0 / 98.9 / 69.0% (above nirnay on four of five tasks).
- **Support routing data is synthetic:** drafted by an AI and judged realistic by a native
  speaker, with labels from the drafter. The other four tasks use public human-labelled data.
- **One run per configuration;** Sarvam sampling isn't deterministic. The 0.8 escalation
  threshold was chosen on the support results; the other tasks are out-of-sample for it.
- **The parser fixes favoured the default:** teaching the parser native-script letters raised the
  default's scores by up to 11 points, and all of its replies were re-scored that way.
- Prices are Sarvam's published rates as of 30 Sep 2026 (₹29.28 input / ₹10.98 cached /
  ₹73.20 output per million tokens).

### Reproducing

```bash
pip install -e ".[dev]"
python experiments/prepare_bench.py                       # builds data/bench/ from the public datasets in data/raw/
python experiments/bench_run.py --system nirnay           # needs SARVAM_API_KEY
python experiments/bench_run.py --system default --splits test
python experiments/bench_analyze.py                       # tables
python experiments/audit_bench.py                         # independent recomputation + significance
python experiments/make_medium_charts.py                  # charts
```

Dataset downloads (place under `data/raw/`): Hinglish-TOP (Apache-2.0), MASSIVE 1.1 (CC BY 4.0),
IndicSentiment (licence not stated on its dataset card), SemEval-2020 SentiMix (CC BY 4.0). The
support-routing set is private.

## Roadmap

- Token-probability (logprobs) confidence via Sarvam's V2 endpoint (currently in beta).
- "Auto mode": detect decision-shaped calls inside an agent and route them through nirnay.
- Async client and a response cache.

## Related work

The escalate-when-unsure idea builds on LLM-cascade research, notably *Mixture-of-Thought
cascades* (ICLR 2024). nirnay applies it to Sarvam using Sarvam's own thinking switch as the two
tiers, so everything stays on one India-hosted provider.

## Development

Copy `.env.example` to `.env` and add your Sarvam API key.

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT. See [LICENSE](https://github.com/Dobbyisus/nirnay/blob/main/LICENSE).
