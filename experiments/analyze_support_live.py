"""Compare live nirnay (votes + escalation) with Sarvam default settings on the support test set.

Uses run_support_live.py output and run_support_baseline.py output (rescored with the current
parser). Also prints the earlier replay estimate for the same threshold, to check the live
run against it.

Run:  python experiments/analyze_support_live.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tasks import SUPPORT_LABELS_DESCRIBED  # noqa: E402

from nirnay.prompt import label_codes, parse_reply  # noqa: E402

RESULTS = ROOT / "data" / "private" / "results"


def rows(name):
    latest = {}
    for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r.get("condition", "described"), r["id"])
        if key not in latest or r.get("error") is None:
            latest[key] = r
    return [r for r in latest.values() if r.get("error") is None]


def strict(r):
    return r["best_guess"] == r["label"]


def lenient(r):
    return r["best_guess"] is not None and r["best_guess"] in {r["label"], r["alt_label"]}


def pct(x):
    return f"{100 * x:5.1f}%"


def p(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    live = rows("support_v0_nirnay_live.jsonl")
    codes = label_codes(SUPPORT_LABELS_DESCRIBED)
    default = {}
    for r in rows("support_v0_default.jsonl"):
        if r["condition"] == "described" and r["split"] == "test":
            code = parse_reply(r["content"], codes)
            r["best_guess"] = codes[code] if code else None
            default[r["id"]] = r
    votes = {
        r["id"]: r
        for r in rows("support_v0_votes.jsonl")
        if r["condition"] == "described" and r["split"] == "test"
    }
    live = [r for r in live if r["id"] in default]
    d = [default[r["id"]] for r in live]
    n = len(live)
    esc = [r for r in live if r["escalated"]]

    print(f"=== Support test set, with label descriptions (n={n}) ===")
    print(f"{'':30}{'Sarvam default settings':>24}{'nirnay':>10}")
    table = [
        ("accuracy (strict)", sum(map(strict, d)) / n, sum(map(strict, live)) / n, pct),
        ("accuracy (alt label ok)", sum(map(lenient, d)) / n, sum(map(lenient, live)) / n, pct),
        (
            "no usable answer",
            sum(r["best_guess"] is None for r in d) / n,
            sum(not r["valid"] for r in live) / n,
            pct,
        ),
        (
            "₹ per 1,000 decisions",
            1000 * statistics.mean(r["cost_inr"] for r in d),
            1000 * statistics.mean(r["cost_inr"] for r in live),
            lambda x: f"₹{x:.2f}",
        ),
        (
            "output tokens / decision",
            statistics.mean(r["output_tokens"] for r in d),
            statistics.mean(r["output_tokens"] for r in live),
            lambda x: f"{x:.0f}",
        ),
        (
            "latency p50 (ms)",
            p([r["latency_ms"] for r in d], 0.5),
            p([r["latency_ms"] for r in live], 0.5),
            lambda x: f"{x:.0f}",
        ),
        (
            "latency p95 (ms)",
            p([r["latency_ms"] for r in d], 0.95),
            p([r["latency_ms"] for r in live], 0.95),
            lambda x: f"{x:.0f}",
        ),
    ]
    for label, a, b, fmt in table:
        print(f"{label:30}{fmt(a):>24}{fmt(b):>10}")
    print(f"{'confidence score':30}{'none':>24}{'yes':>10}")

    print(f"\nescalated to thinking: {len(esc)} ({pct(len(esc) / n)})")
    if esc:
        agree = sum(r["escalation_answer"] == r["vote_winner"] for r in esc)
        no_answer = sum(r["escalation_answer"] is None for r in esc)
        print(
            f"   thinking agreed with the vote winner: {agree}; changed it: "
            f"{len(esc) - agree - no_answer}; gave no usable answer (kept votes): {no_answer}"
        )
        rest = [r for r in live if not r["escalated"]]
        esc_acc = sum(map(strict, esc)) / len(esc)
        rest_acc = sum(map(strict, rest)) / len(rest) if rest else 0
        print(f"   accuracy on escalated: {pct(esc_acc)};  on the rest: {pct(rest_acc)}")
    kept = [r for r in live if not r["escalated"]]
    if kept:
        conf = [r["confidence"] for r in kept]
        print(
            f"non-escalated: mean reported confidence {statistics.mean(conf):.2f}, "
            f"accuracy {pct(sum(map(strict, kept)) / len(kept))}"
        )

    # Replay estimate from the two separate runs, same threshold, same items.
    t = live[0]["escalate_below"]
    replay_acc, replay_cost = [], []
    for r in live:
        v, dd = votes.get(r["id"]), default[r["id"]]
        if v is None:
            continue
        e = v["raw_confidence"] < t
        replay_acc.append(strict(dd) if e else strict(v))
        replay_cost.append(v["cost_inr"] + (dd["cost_inr"] if e else 0))
    if replay_acc:
        acc = sum(replay_acc) / len(replay_acc)
        cost = 1000 * statistics.mean(replay_cost)
        print(f"\nreplay estimate (earlier separate runs, < {t}): {pct(acc)}, ₹{cost:.2f}/1k")

    for field in ("language", "difficulty"):
        groups = defaultdict(lambda: ([], []))
        for r in live:
            groups[r[field]][0].append(strict(default[r["id"]]))
            groups[r[field]][1].append(strict(r))
        for g, (da, na) in sorted(groups.items()):
            print(
                f"   {g:10} n={len(da):3}  Sarvam default {pct(sum(da) / len(da))}  "
                f"nirnay {pct(sum(na) / len(na))}"
            )
    print(f"\nlive run total cost ₹{sum(r['cost_inr'] for r in live):.2f}")


if __name__ == "__main__":
    main()
