"""Analyse results from run_support_confidence.py.

For each condition (names only / with descriptions): fit an isotonic calibrator on the dev
split, then report on the test split: accuracy, how honest the raw and calibrated confidence
are (ECE), whether confidence separates right from wrong (AUROC), how often the model is
confidently wrong, the accuracy you'd get by escalating low-confidence decisions, and cost.

Run:  python experiments/analyze_support_confidence.py
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nirnay.calibration import (  # noqa: E402
    IsotonicCalibrator,
    expected_calibration_error,
    reliability_bins,
    risk_coverage,
)

RESULTS = ROOT / "data" / "private" / "results"
RUN = RESULTS / "support_v0_votes.jsonl"
THRESHOLDS = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]


def auroc(conf, correct):
    """Probability that a random correct decision has higher confidence than a random wrong
    one (ties count half). 0.5 = confidence carries no signal; 1.0 = perfect separation."""
    pos = [c for c, ok in zip(conf, correct, strict=True) if ok]
    neg = [c for c, ok in zip(conf, correct, strict=True) if not ok]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def pct(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def analyse(rows, name):
    ok_rows = [r for r in rows if r.get("error") is None]
    dev = [r for r in ok_rows if r["split"] == "dev" and r["raw_confidence"] is not None]
    test = [r for r in ok_rows if r["split"] == "test"]
    errors = len(rows) - len(ok_rows)

    def strict(r):
        return r["best_guess"] == r["label"]

    def lenient(r):
        return r["best_guess"] is not None and r["best_guess"] in {r["label"], r["alt_label"]}

    cal = IsotonicCalibrator().fit([r["raw_confidence"] for r in dev], [strict(r) for r in dev])
    cal.save(RESULTS / f"support_v0_calibrator_{name}.json")

    scored = [r for r in test if r["raw_confidence"] is not None]
    raw = [r["raw_confidence"] for r in scored]
    calib = [cal(c) for c in raw]
    right = [strict(r) for r in scored]

    out = {
        "condition": name,
        "test_items": len(test),
        "api_errors": errors,
        "invalid_decisions": sum(not r["valid"] for r in test),
        "accuracy_strict": sum(map(strict, test)) / len(test),
        "accuracy_lenient": sum(map(lenient, test)) / len(test),
        "ece_raw": expected_calibration_error(raw, right),
        "ece_calibrated": expected_calibration_error(calib, right),
        "auroc": auroc(raw, right),
        "confidently_wrong": sum(c >= 0.9 and not ok for c, ok in zip(raw, right, strict=True))
        / len(scored),
        "wrong_that_were_confident": (
            sum(c >= 0.9 and not ok for c, ok in zip(raw, right, strict=True))
            / max(1, sum(not ok for ok in right))
        ),
        "risk_coverage_raw": risk_coverage(raw, right, THRESHOLDS),
        "calibrator": dict(zip(cal.xs, cal.ys, strict=True)),
        "reliability_raw": [b.__dict__ for b in reliability_bins(raw, right)],
        "by_language": {},
        "by_difficulty": {},
        "top_confusions": Counter(
            f"{r['label']}→{r['best_guess']}" for r in test if not strict(r)
        ).most_common(8),
        "cost_per_1000_inr": 1000 * statistics.mean(r["cost_inr"] for r in test),
        "input_tokens_mean": statistics.mean(r["input_tokens"] for r in test),
        "cached_share": sum(r["cached_input_tokens"] for r in test)
        / sum(r["input_tokens"] for r in test),
        "latency_p50_ms": percentile([r["latency_ms"] for r in test], 0.5),
        "latency_p95_ms": percentile([r["latency_ms"] for r in test], 0.95),
    }
    for key, field in [("by_language", "language"), ("by_difficulty", "difficulty")]:
        groups = defaultdict(list)
        for r in scored:
            groups[r[field]].append(r)
        for g, rs in sorted(groups.items()):
            rr = [strict(r) for r in rs]
            cc = [r["raw_confidence"] for r in rs]
            out[key][g] = {
                "n": len(rs),
                "accuracy": sum(rr) / len(rs),
                "mean_raw_confidence": statistics.mean(cc),
                "ece_raw": expected_calibration_error(cc, rr),
                "confidently_wrong": sum(c >= 0.9 and not ok for c, ok in zip(cc, rr, strict=True))
                / len(rs),
            }
    return out


def report(s):
    print(f"\n=== {s['condition'].upper()} (test set, n={s['test_items']}) ===")
    print(
        f"accuracy        strict {pct(s['accuracy_strict'])}   lenient (alt label ok) "
        f"{pct(s['accuracy_lenient'])}"
    )
    print(f"api errors {s['api_errors']}, invalid decisions {s['invalid_decisions']}")
    print(
        f"ECE             raw {s['ece_raw']:.3f}  →  calibrated {s['ece_calibrated']:.3f}"
        "   (0 = perfectly honest)"
    )
    au = s["auroc"]
    print(
        f"AUROC           {'n/a' if au is None else f'{au:.3f}'}"
        "   (0.5 = confidence useless, 1.0 = perfectly separates right/wrong)"
    )
    print(
        f"confidently wrong (≥90% votes, wrong): {pct(s['confidently_wrong'])} of decisions; "
        f"{pct(s['wrong_that_were_confident'])} of all mistakes"
    )
    print(
        "calibrator (raw vote share → calibrated):",
        ", ".join(f"{x:.1f}→{y:.2f}" for x, y in s["calibrator"].items()),
    )
    print("escalate below raw threshold → answered share / accuracy on answered:")
    for t, cov, acc in s["risk_coverage_raw"]:
        print(f"   t={t:.1f}: answered {pct(cov)}, accuracy {pct(acc)}, escalated {pct(1 - cov)}")
    for key in ("by_language", "by_difficulty"):
        for g, v in s[key].items():
            print(
                f"   {g:10} n={v['n']:3}  acc {pct(v['accuracy'])}  mean conf "
                f"{v['mean_raw_confidence']:.2f}  ECE {v['ece_raw']:.3f}  "
                f"conf-wrong {pct(v['confidently_wrong'])}"
            )
    print("top confusions (true→guess):", s["top_confusions"])
    print(
        f"cost ₹{s['cost_per_1000_inr']:.2f}/1k  input {s['input_tokens_mean']:.0f} tok "
        f"({pct(s['cached_share'])} cached)  latency p50 {s['latency_p50_ms']:.0f} ms, "
        f"p95 {s['latency_p95_ms']:.0f} ms"
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(line) for line in RUN.read_text(encoding="utf-8").splitlines()]
    latest = {}
    for r in rows:  # keep the last attempt per (condition, id), preferring successes
        key = (r["condition"], r["id"])
        if key not in latest or r.get("error") is None:
            latest[key] = r
    summary = {}
    for cond in ("names", "described"):
        cond_rows = [r for (c, _), r in latest.items() if c == cond]
        if cond_rows:
            summary[cond] = analyse(cond_rows, cond)
            report(summary[cond])
    (RESULTS / "support_v0_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
