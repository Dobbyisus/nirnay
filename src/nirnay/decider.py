"""The Decider: ask the model several times in one call and use the votes as confidence."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .client import V1_CHAT, SarvamClient
from .pricing import cost_inr
from .prompt import build_messages, label_codes, parse_reply

# Models that accept `reasoning_effort`. Others (e.g. sarvam-105b-conversations) reject it,
# so it is left out of their requests.
REASONING_MODELS = frozenset({"sarvam-105b"})


@dataclass(frozen=True)
class Usage:
    """Tokens and money spent on one decision, including any retry."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_inr: float | None = None
    calls: int = 0


@dataclass(frozen=True)
class Decision:
    """The outcome of :meth:`Decider.decide`.

    ``choice`` is what the caller should act on: the winning label, or ``None`` when the
    Decider abstained (confidence below ``threshold``). If no reply was usable even after a
    retry, ``choice`` is the configured ``fallback`` and ``valid`` is ``False``.
    ``best_guess`` is always the winning label (or ``None`` if nothing was usable).
    """

    choice: str | None
    best_guess: str | None
    confidence: float | None
    votes: dict[str, int]
    samples: int
    valid: bool
    abstained: bool
    invalid_samples: int
    retried: bool
    usage: Usage
    latency_ms: float
    model: str
    raw_replies: list[str | None] = field(repr=False, default_factory=list)

    @property
    def probabilities(self) -> dict[str, float]:
        """Share of samples that voted for each label (unusable replies count as no vote)."""
        if not self.samples:
            return {}
        return {label: count / self.samples for label, count in self.votes.items()}


class Decider:
    """Pick one label from a fixed list, with a vote-based confidence.

    One API call asks the model for ``samples`` independent answers (Sarvam's ``n``
    parameter), so the prompt is billed once and only the tiny answers are multiplied.
    Confidence is the share of samples that agree with the winner. Replies that can't be
    matched to a choice count as votes for nothing, which lowers the confidence.

    Args:
        model: Sarvam model ID.
        samples: answers per decision (1–128). 1 turns voting off: no confidence, and the
            default temperature becomes 0.
        temperature: sampling randomness. Voting needs some (default 1.0 when samples > 1).
        threshold: if set, decisions with confidence below it abstain (``choice=None``).
        fallback: returned as ``choice`` when no usable reply came back, even after a retry.
        reasoning_effort: ``None`` (default) turns hidden reasoning off, which is much
            faster and cheaper for this kind of question. Pass "low"/"medium"/"high" to
            turn it on. Only sent to models that accept it.
        max_tokens: cap per answer. The answer is one letter; a little slack absorbs a
            leading newline or trailing punctuation.
        client: a configured :class:`SarvamClient`; one is created from ``SARVAM_API_KEY``
            if omitted.
    """

    def __init__(
        self,
        model: str = "sarvam-105b",
        *,
        samples: int = 10,
        temperature: float | None = None,
        threshold: float | None = None,
        fallback: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int = 4,
        client: SarvamClient | None = None,
        endpoint: str = V1_CHAT,
    ):
        if not 1 <= samples <= 128:
            raise ValueError("samples must be between 1 and 128")
        if temperature is None:
            temperature = 1.0 if samples > 1 else 0.0
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if threshold is not None:
            if samples == 1:
                raise ValueError("threshold needs samples > 1 (one sample has no confidence)")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("threshold must be between 0 and 1")
        self.model = model
        self.samples = samples
        self.temperature = temperature
        self.threshold = threshold
        self.fallback = fallback
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.endpoint = endpoint
        self._own_client = client is None
        self.client = client or SarvamClient()

    def decide(self, question: str, choices: Sequence[str], context: str) -> Decision:
        codes = label_codes(choices)
        start = time.perf_counter()
        totals = Counter()

        replies = self._sample(build_messages(question, codes, context), totals)
        picked = [parse_reply(r, codes) for r in replies]
        retried = False
        if not any(picked):
            retried = True
            replies = self._sample(build_messages(question, codes, context, strict=True), totals)
            picked = [parse_reply(r, codes) for r in replies]

        n = len(replies)
        counts = Counter(p for p in picked if p is not None)
        # Ties go to the choice listed first; the low confidence already flags them.
        order = list(codes)
        ranked = sorted(counts, key=lambda c: (-counts[c], order.index(c)))

        usage = Usage(
            input_tokens=totals["input"],
            cached_input_tokens=totals["cached"],
            output_tokens=totals["output"],
            cost_inr=cost_inr(self.model, totals["input"], totals["cached"], totals["output"]),
            calls=totals["calls"],
        )
        # Time spent deliberately pacing calls for rate limits isn't the model's latency.
        latency_ms = (time.perf_counter() - start) * 1000 - totals["paced_ms"]

        if not ranked:
            return Decision(
                choice=self.fallback,
                best_guess=None,
                confidence=None,
                votes={},
                samples=n,
                valid=False,
                abstained=False,
                invalid_samples=n,
                retried=retried,
                usage=usage,
                latency_ms=latency_ms,
                model=self.model,
                raw_replies=replies,
            )

        best = codes[ranked[0]]
        confidence = counts[ranked[0]] / n if self.samples > 1 else None
        abstained = (
            self.threshold is not None and confidence is not None and confidence < self.threshold
        )
        return Decision(
            choice=None if abstained else best,
            best_guess=best,
            confidence=confidence,
            votes={codes[c]: counts[c] for c in ranked},
            samples=n,
            valid=True,
            abstained=abstained,
            invalid_samples=n - sum(counts.values()),
            retried=retried,
            usage=usage,
            latency_ms=latency_ms,
            model=self.model,
            raw_replies=replies,
        )

    def _sample(self, messages: list[dict[str, str]], totals: Counter) -> list[str | None]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.samples > 1:
            body["n"] = self.samples
        if self.model in REASONING_MODELS:
            body["reasoning_effort"] = self.reasoning_effort  # None → JSON null → reasoning off
        data = self.client.chat(body, endpoint=self.endpoint)
        totals["paced_ms"] += self.client.last_paced_s * 1000

        u = data.get("usage") or {}
        totals["input"] += u.get("prompt_tokens") or 0
        totals["output"] += u.get("completion_tokens") or 0
        totals["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        totals["calls"] += 1
        return [(c.get("message") or {}).get("content") for c in data.get("choices") or []]

    def close(self) -> None:
        if self._own_client:
            self.client.close()

    def __enter__(self) -> Decider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
