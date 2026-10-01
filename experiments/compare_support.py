"""Head-to-head on the support test split: Sarvam default settings vs nirnay.

Needs results from run_support_baseline.py and run_support_confidence.py, and the calibrators
saved by analyze_support_confidence.py (run that first).

Run:  python experiments/compare_support.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tasks import SUPPORT_LABELS, SUPPORT_LABELS_DESCRIBED  # noqa: E402

from nirnay.calibration import IsotonicCalibrator  # noqa: E402
from nirnay.prompt import label_codes, parse_reply  # noqa: E402

CONDITIONS = {"names": SUPPORT_LABELS, "described": SUPPORT_LABELS_DESCRIBED}

RESULTS = ROOT / "data" / "private" / "results"


def rescore(r):
    """Re-parse a default-settings reply with the current parser (answers are saved raw)."""
    if "content" in r:
        codes = label_codes(CONDITIONS[r["condition"]])
        code = parse_reply(r["content"], codes)
        r["best_guess"], r["valid"] = (codes[code] if code else None), code is not None
    return r


def load(name):
    latest = {}
    for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r["condition"], r["id"])
        if key not in latest or r.get("error") is None:
            latest[key] = r
    return [rescore(r) for r in latest.values() if r["split"] == "test" and r.get("error") is None]


def strict(r):
    return r["best_guess"] == r["label"]


def lenient(r):
    return r["best_guess"] is not None and r["best_guess"] in {r["label"], r["alt_label"]}


def pct(x):
    return f"{100 * x:5.1f}%"


def p(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarise(rows):
    return {
        "n": len(rows),
        "strict": sum(map(strict, rows)) / len(rows),
        "lenient": sum(map(lenient, rows)) / len(rows),
        "invalid": sum(not r["valid"] for r in rows) / len(rows),
        "cost_1k": 1000 * statistics.mean(r["cost_inr"] for r in rows),
        "out_tokens": statistics.mean(r["output_tokens"] for r in rows),
        "p50": p([r["latency_ms"] for r in rows], 0.5),
        "p95": p([r["latency_ms"] for r in rows], 0.95),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    default, votes = load("support_v0_default.jsonl"), load("support_v0_votes.jsonl")
    for cond in ("names", "described"):
        d = [r for r in default if r["condition"] == cond]
        v = [r for r in votes if r["condition"] == cond]
        if not d or not v:
            continue
        sd, sv = summarise(d), summarise(v)
        print(f"\n=== {cond.upper()} — test set (default n={sd['n']}, nirnay n={sv['n']}) ===")
        print(f"{'':28}{'Sarvam default':>16}{'nirnay':>12}")
        for key, label, fmt in [
            ("strict", "accuracy (strict)", pct),
            ("lenient", "accuracy (alt label ok)", pct),
            ("invalid", "unusable answers", pct),
            ("cost_1k", "₹ per 1,000 decisions", lambda x: f"₹{x:.2f}"),
            ("out_tokens", "output tokens / decision", lambda x: f"{x:.0f}"),
            ("p50", "latency p50 (ms)", lambda x: f"{x:.0f}"),
            ("p95", "latency p95 (ms)", lambda x: f"{x:.0f}"),
        ]:
            print(f"{label:28}{fmt(sd[key]):>16}{fmt(sv[key]):>12}")
        print(f"{'confidence / escalation':28}{'none':>16}{'yes':>12}")
        reasoning = statistics.mean(r["reasoning_chars"] for r in d)
        cut_off = sum(r["finish_reason"] != "stop" for r in d)
        print(f"default reasoning chars, mean {reasoning:.0f}; finish_reason≠stop: {cut_off}")

        cal_path = RESULTS / f"support_v0_calibrator_{cond}.json"
        if cal_path.exists():
            cal = IsotonicCalibrator.load(cal_path)
            print("nirnay with escalation (calibrated confidence threshold):")
            for t in (0.7, 0.8, 0.9):
                kept = [r for r in v if cal(r["raw_confidence"]) >= t]
                acc = sum(map(strict, kept)) / len(kept) if kept else 0
                print(
                    f"   escalate below {t:.1f}: {pct(1 - len(kept) / len(v))} to humans, "
                    f"accuracy on the rest {pct(acc)}"
                )

        by_id = {r["id"]: r for r in d}
        both = [(by_id[r["id"]], r) for r in v if r["id"] in by_id]
        only_default = sum(strict(a) and not strict(b) for a, b in both)
        only_nirnay = sum(strict(b) and not strict(a) for a, b in both)
        print(f"right only with default: {only_default}   right only with nirnay: {only_nirnay}")

        # Cascade: nirnay answers when enough votes agree; otherwise the same message goes to
        # the default (thinking-on) call. Both answers exist for every message, so this is
        # computed exactly from the logged results: cost and latency add up for escalated ones.
        print("cascade: nirnay first, escalate to Sarvam default when votes agree less than…")
        base_cost = statistics.mean(a["cost_inr"] for a, _ in both) * 1000
        base_acc = sum(strict(a) for a, _ in both) / len(both)
        print(f"   (default alone: accuracy {pct(base_acc)}, ₹{base_cost:.2f}/1k)")
        for t in (0.6, 0.8, 0.9, 1.0):
            esc = [b["raw_confidence"] < t for _, b in both]
            final = [strict(a) if e else strict(b) for (a, b), e in zip(both, esc, strict=True)]
            cost = [
                b["cost_inr"] + (a["cost_inr"] if e else 0)
                for (a, b), e in zip(both, esc, strict=True)
            ]
            lat = [
                b["latency_ms"] + (a["latency_ms"] if e else 0)
                for (a, b), e in zip(both, esc, strict=True)
            ]
            print(
                f"   < {t:.1f} ({round(10 * t)}/10 votes): escalated {pct(sum(esc) / len(esc))}, "
                f"accuracy {pct(sum(final) / len(final))}, ₹{1000 * statistics.mean(cost):.2f}/1k "
                f"({pct(1000 * statistics.mean(cost) / base_cost)} of default), "
                f"latency p50 {p(lat, 0.5):.0f} ms, p95 {p(lat, 0.95):.0f} ms"
            )
        for field in ("language", "difficulty"):
            groups = defaultdict(lambda: ([], []))
            for a, b in both:
                groups[a[field]][0].append(strict(a))
                groups[a[field]][1].append(strict(b))
            for g, (da, na) in sorted(groups.items()):
                print(
                    f"   {g:10} n={len(da):3}  default {pct(sum(da) / len(da))}  "
                    f"nirnay {pct(sum(na) / len(na))}"
                )


if __name__ == "__main__":
    main()
