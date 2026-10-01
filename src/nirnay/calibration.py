"""Turn raw confidence (vote share) into calibrated confidence, and measure calibration.

A confidence is *calibrated* when, among decisions given confidence ~0.8, about 80% are
correct. Raw vote shares usually aren't: "10 of 10 votes" is often right less than 100% of
the time. :class:`IsotonicCalibrator` learns the mapping from labelled examples:

    cal = IsotonicCalibrator().fit(raw_confidences, was_correct)
    cal(0.9)  # e.g. 0.82: how often decisions with 90% of the votes were actually right

Isotonic regression only assumes that more agreement never means *less* accuracy, which suits
coarse vote shares (0.1, 0.2, …, 1.0) and small labelled sets.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class IsotonicCalibrator:
    """Monotone mapping from raw confidence to observed accuracy (pool-adjacent-violators)."""

    def __init__(self, xs: Sequence[float] = (), ys: Sequence[float] = ()):
        self.xs = list(xs)  # sorted raw-confidence knots
        self.ys = list(ys)  # calibrated value at each knot (non-decreasing)

    def fit(self, confidences: Sequence[float], correct: Sequence[bool]) -> IsotonicCalibrator:
        if len(confidences) != len(correct):
            raise ValueError("confidences and correct must have the same length")
        if not confidences:
            raise ValueError("need at least one example to fit")
        # One block per distinct confidence value: [x, total correct, count].
        totals: dict[float, list[float]] = {}
        for c, ok in zip(confidences, correct, strict=True):
            block = totals.setdefault(float(c), [0.0, 0.0])
            block[0] += 1.0 if ok else 0.0
            block[1] += 1.0
        blocks = [[x, s, n] for x, (s, n) in sorted(totals.items())]

        # Pool adjacent blocks while accuracy decreases left to right.
        merged: list[list[float]] = []
        for x, s, n in blocks:
            merged.append([x, s, n, x])  # x_lo, sum, count, x_hi
            while len(merged) > 1 and merged[-2][1] / merged[-2][2] > merged[-1][1] / merged[-1][2]:
                lo, s1, n1, _ = merged[-2]
                _, s2, n2, hi = merged.pop()
                merged[-1] = [lo, s1 + s2, n1 + n2, hi]

        self.xs, self.ys = [], []
        for lo, s, n, hi in merged:
            for x in (lo, hi) if hi != lo else (lo,):
                self.xs.append(x)
                self.ys.append(s / n)
        return self

    def __call__(self, confidence: float) -> float:
        """Calibrated confidence, interpolating linearly between knots and clamping outside."""
        if not self.xs:
            raise RuntimeError("calibrator is not fitted")
        if confidence <= self.xs[0]:
            return self.ys[0]
        if confidence >= self.xs[-1]:
            return self.ys[-1]
        i = bisect_right(self.xs, confidence)
        x0, x1, y0, y1 = self.xs[i - 1], self.xs[i], self.ys[i - 1], self.ys[i]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (confidence - x0) / (x1 - x0)

    def to_dict(self) -> dict:
        return {"type": "isotonic", "xs": self.xs, "ys": self.ys}

    @classmethod
    def from_dict(cls, d: dict) -> IsotonicCalibrator:
        if d.get("type") != "isotonic":
            raise ValueError(f"not an isotonic calibrator: {d.get('type')!r}")
        return cls(d["xs"], d["ys"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> IsotonicCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> list[Bin]:
    """Equal-width bins over [0, 1]; empty bins are omitted. The last bin includes 1.0."""
    groups: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct, strict=True):
        groups[min(int(c * n_bins), n_bins - 1)].append((c, ok))
    return [
        Bin(
            lo=i / n_bins,
            hi=(i + 1) / n_bins,
            count=len(g),
            mean_confidence=sum(c for c, _ in g) / len(g),
            accuracy=sum(ok for _, ok in g) / len(g),
        )
        for i, g in enumerate(groups)
        if g
    ]


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Average |confidence − accuracy| over bins, weighted by bin size (0 = perfect)."""
    total = len(confidences)
    if not total:
        raise ValueError("need at least one example")
    bins = reliability_bins(confidences, correct, n_bins)
    return sum(b.count / total * abs(b.mean_confidence - b.accuracy) for b in bins)


def risk_coverage(
    confidences: Sequence[float], correct: Sequence[bool], thresholds: Sequence[float]
) -> list[tuple[float, float, float | None]]:
    """For each threshold: (threshold, share of decisions answered, accuracy on those).

    Decisions below the threshold are the ones that would be escalated to a human.
    """
    total = len(confidences)
    rows = []
    for t in thresholds:
        kept = [ok for c, ok in zip(confidences, correct, strict=True) if c >= t]
        rows.append(
            (t, len(kept) / total if total else 0.0, sum(kept) / len(kept) if kept else None)
        )
    return rows
