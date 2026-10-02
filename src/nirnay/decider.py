"""The Decider: ask the model several times in one call, use the votes as confidence, and
escalate unsure decisions to the model's thinking mode."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .client import V1_CHAT, SarvamClient
from .pricing import cost_inr
from .prompt import Choices, build_messages, label_codes, label_descriptions, parse_reply

# Models that accept `reasoning_effort`. Others (e.g. sarvam-105b-conversations) reject it,
# so it is left out of their requests.
REASONING_MODELS = frozenset({"sarvam-105b"})

DEFAULT_ESCALATE_BELOW = 0.8
_AUTO: Any = object()


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
    ``best_guess`` is always the final answer (or ``None`` if nothing was usable).
    ``raw_confidence`` is the vote winner's share; ``confidence`` is that value passed through
    the Decider's calibrator, or the same as ``raw_confidence`` if there is none.

    When the votes were too split, the decision was ``escalated`` to the model's thinking
    mode: ``choice`` is then that answer (or the vote winner, if thinking gave no usable
    answer), ``escalation_answer`` holds what thinking said, and ``confidence`` is ``None``
    because only the voting path has a calibrated confidence. Escalated decisions never
    abstain.
    """

    choice: str | None
    best_guess: str | None
    confidence: float | None
    raw_confidence: float | None
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
    escalated: bool = False
    escalation_answer: str | None = None
    escalation_reply: str | None = field(repr=False, default=None)

    @property
    def probabilities(self) -> dict[str, float]:
        """Share of samples that voted for each label (unusable replies count as no vote)."""
        if not self.samples:
            return {}
        return {label: count / self.samples for label, count in self.votes.items()}


class Decider:
    """Pick one label from a fixed list, with a vote-based confidence and escalation.

    One API call asks the model for ``samples`` independent answers (Sarvam's ``n``
    parameter), so the prompt is billed once and only the tiny answers are multiplied.
    Confidence is the share of samples that agree with the winner. Replies that can't be
    matched to a choice count as votes for nothing, which lowers the confidence.

    When fewer than ``escalate_below`` of the votes agree, the same question is sent once
    more with the model's default settings (thinking on), and that answer is used. Most
    decisions stay on the fast, cheap voting path; only unsure ones pay for thinking.

    Args:
        model: Sarvam model ID.
        samples: answers per decision (1–128). 1 turns voting off: no confidence, and the
            default temperature becomes 0.
        temperature: sampling randomness. Voting needs some (default 1.0 when samples > 1).
        escalate_below: escalate when the winner's vote share is below this (0.8 = fewer
            than 8 of 10 agree). Defaults to 0.8 for models with thinking and off otherwise;
            pass ``None`` to turn escalation off.
        escalation_max_tokens: token cap for the thinking call; ``None`` uses the server
            default (2048 for sarvam-105b), which thinking counts against.
        threshold: if set, non-escalated decisions with confidence below it abstain
            (``choice=None``). Applied to the calibrated confidence when a calibrator is set.
        calibrator: maps raw vote share to calibrated confidence, e.g. a fitted
            :class:`~nirnay.calibration.IsotonicCalibrator`. Any ``float -> float`` callable works.
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
        escalate_below: float | None = _AUTO,
        escalation_max_tokens: int | None = None,
        threshold: float | None = None,
        calibrator: Callable[[float], float] | None = None,
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
        if calibrator is not None and samples == 1:
            raise ValueError("calibrator needs samples > 1 (one sample has no confidence)")
        if escalate_below is _AUTO:
            thinking = model in REASONING_MODELS and samples > 1
            escalate_below = DEFAULT_ESCALATE_BELOW if thinking else None
        if escalate_below is not None:
            if samples == 1:
                raise ValueError("escalate_below needs samples > 1 (one sample has no votes)")
            if not 0.0 < escalate_below <= 1.0:
                raise ValueError("escalate_below must be in (0, 1]")
            if model not in REASONING_MODELS:
                raise ValueError(f"escalation needs a model with thinking; {model!r} has none")
        self.escalate_below = escalate_below
        self.escalation_max_tokens = escalation_max_tokens
        self.calibrator = calibrator
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

    def decide(self, question: str, choices: Choices, context: str) -> Decision:
        """Pick one of ``choices`` for ``context``.

        ``choices`` is a list of labels, or a mapping of label → short description for labels
        whose meaning isn't obvious from the name. Descriptions cost extra input tokens (billed
        at the cached rate after the first call), so add them where they help.
        """
        codes = label_codes(choices)
        descriptions = label_descriptions(choices)
        start = time.perf_counter()
        totals = Counter()

        messages = build_messages(question, codes, context, descriptions=descriptions)
        replies = self._sample(messages, totals)
        picked = [parse_reply(r, codes) for r in replies]
        retried = False
        if not any(picked):
            retried = True
            messages = build_messages(
                question, codes, context, descriptions=descriptions, strict=True
            )
            replies = self._sample(messages, totals)
            picked = [parse_reply(r, codes) for r in replies]

        n = len(replies)
        counts = Counter(p for p in picked if p is not None)
        # Ties go to the choice listed first; the low confidence already flags them.
        order = list(codes)
        ranked = sorted(counts, key=lambda c: (-counts[c], order.index(c)))
        winner = codes[ranked[0]] if ranked else None
        raw = counts[ranked[0]] / n if ranked and self.samples > 1 else None

        # Escalate split (or entirely unusable) votes to the model's thinking mode, using the
        # original prompt. If thinking gives no usable answer, keep the vote winner.
        escalated, escalation_answer, content = False, None, None
        if self.escalate_below is not None and (raw is None or raw < self.escalate_below):
            escalated = True
            content = self._think(
                build_messages(question, codes, context, descriptions=descriptions), totals
            )
            code = parse_reply(content, codes)
            escalation_answer = codes[code] if code else None

        usage = Usage(
            input_tokens=totals["input"],
            cached_input_tokens=totals["cached"],
            output_tokens=totals["output"],
            cost_inr=cost_inr(self.model, totals["input"], totals["cached"], totals["output"]),
            calls=totals["calls"],
        )
        # Time spent deliberately pacing calls for rate limits isn't the model's latency.
        latency_ms = (time.perf_counter() - start) * 1000 - totals["paced_ms"]

        final = escalation_answer or winner
        if escalated:
            confidence, abstained = None, False
        else:
            confidence = self.calibrator(raw) if raw is not None and self.calibrator else raw
            abstained = (
                self.threshold is not None
                and confidence is not None
                and confidence < self.threshold
            )
        choice = self.fallback if final is None else final
        if abstained:
            choice = None
        return Decision(
            choice=choice,
            best_guess=final,
            confidence=confidence,
            raw_confidence=raw,
            votes={codes[c]: counts[c] for c in ranked},
            samples=n,
            valid=final is not None,
            abstained=abstained,
            invalid_samples=n - sum(counts.values()),
            retried=retried,
            usage=usage,
            latency_ms=latency_ms,
            model=self.model,
            raw_replies=replies,
            escalated=escalated,
            escalation_answer=escalation_answer,
            escalation_reply=content,
        )

    def _think(self, messages: list[dict[str, str]], totals: Counter) -> str | None:
        """One answer with the model's default settings (thinking on)."""
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.escalation_max_tokens is not None:
            body["max_tokens"] = self.escalation_max_tokens
        data = self.client.chat(body, endpoint=self.endpoint)
        self._account(data, totals)
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content")

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
        self._account(data, totals)
        return [(c.get("message") or {}).get("content") for c in data.get("choices") or []]

    def _account(self, data: dict[str, Any], totals: Counter) -> None:
        totals["paced_ms"] += self.client.last_paced_s * 1000
        u = data.get("usage") or {}
        totals["input"] += u.get("prompt_tokens") or 0
        totals["output"] += u.get("completion_tokens") or 0
        totals["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        totals["calls"] += 1

    def close(self) -> None:
        if self._own_client:
            self.client.close()

    def __enter__(self) -> Decider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
