"""Summarise every benchmark: Sarvam default settings vs nirnay, per task, condition and language.

Reads the public-task results from bench_run.py and the support-routing results from the
private folder (if present). For each task × condition it reports accuracy, unusable answers,
cost, latency and escalation, and checks nirnay's confidence: a calibrator is fitted on the dev
split (vote share → was the vote winner right?) and its honesty (ECE) is measured on the
non-escalated test decisions. Default replies are re-scored with the current parser.

Writes experiments/results/bench_summary.json (aggregates only, no message text).

Run:  python experiments/bench_analyze.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tasks import choices  # noqa: E402

from nirnay.calibration import IsotonicCalibrator, expected_calibration_error  # noqa: E402
from nirnay.prompt import label_codes, parse_reply  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
PRIVATE = ROOT / "data" / "private" / "results"
ORDER = ["support", "top", "massive", "reviews", "tweets"]
TITLES = {
    "support": "Support routing (Hindi, Hinglish; 11 labels)",
    "top": "Voice-assistant domain (English vs Hinglish, same requests; 8 labels)",
    "massive": "Voice-assistant scenario (English, Hindi, Bengali, Tamil; 18 labels)",
    "reviews": "Product-review sentiment (English, Hindi, Bengali, Tamil; 2 labels)",
    "tweets": "Tweet sentiment (Hinglish, code-mixed; 3 labels)",
}


def read(path, **defaults):
    if not path.exists():
        return []
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = {**defaults, **json.loads(line)}
        key = (r["condition"], r["id"])
        if key not in latest or r.get("error") is None:
            latest[key] = r
    return [r for r in latest.values() if r.get("error") is None]


def rescore_default(r, task):
    codes = label_codes(choices(task, r["condition"]))
    code = parse_reply(r.get("content"), codes)
    r["best_guess"] = codes[code] if code else None
    return r


def rescore_nirnay(r, task):
    """Re-read a saved thinking reply with the current parser (as nirnay would today): the
    answer is the thinking answer, or the vote winner if thinking gave no usable answer."""
    if r.get("escalated") and r.get("escalation_reply") is not None:
        codes = label_codes(choices(task, r["condition"]))
        code = parse_reply(r["escalation_reply"], codes)
        r["escalation_answer"] = codes[code] if code else None
        r["best_guess"] = r["escalation_answer"] or r["vote_winner"]
    return r


def load():
    """{task: {"nirnay": rows, "default": rows, "calib": dev rows with raw_confidence}}"""
    data = {}
    for task in ORDER[1:]:
        nirnay = [rescore_nirnay(r, task) for r in read(RESULTS / f"bench_{task}_nirnay.jsonl")]
        default = [rescore_default(r, task) for r in read(RESULTS / f"bench_{task}_default.jsonl")]
        calib = [
            {**r, "winner": r["vote_winner"]}
            for r in nirnay
            if r["split"] == "dev" and r["raw_confidence"] is not None
        ]
        data[task] = {"nirnay": nirnay, "default": default, "calib": calib}
    support_nirnay = read(PRIVATE / "support_v0_nirnay_live.jsonl", condition="described") + read(
        PRIVATE / "support_v0_nirnay_live_names.jsonl", condition="names"
    )
    if support_nirnay:
        default = [
            rescore_default(r, "support") for r in read(PRIVATE / "support_v0_default.jsonl")
        ]
        calib = [
            {**r, "winner": r["best_guess"]}  # votes-only run: best guess = vote winner
            for r in read(PRIVATE / "support_v0_votes.jsonl")
            if r["split"] == "dev" and r["raw_confidence"] is not None
        ]
        data["support"] = {"nirnay": support_nirnay, "default": default, "calib": calib}
    return data


def p(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def acc(rows):
    return sum(r["best_guess"] == r["label"] for r in rows) / len(rows) if rows else None


def system_stats(rows):
    if not rows:
        return None
    return {
        "n": len(rows),
        "accuracy": acc(rows),
        "unusable": sum(r["best_guess"] is None for r in rows) / len(rows),
        "cost_per_1k": 1000 * statistics.mean(r["cost_inr"] for r in rows),
        "latency_p50": p([r["latency_ms"] for r in rows], 0.5),
        "latency_p95": p([r["latency_ms"] for r in rows], 0.95),
        "output_tokens": statistics.mean(r["output_tokens"] for r in rows),
    }


def analyse(task, d, condition):
    nir = [r for r in d["nirnay"] if r["condition"] == condition and r["split"] == "test"]
    dft = [r for r in d["default"] if r["condition"] == condition and r["split"] == "test"]
    # Compare both systems on exactly the same messages (matters while runs are in progress).
    if dft:
        both = {r["id"] for r in nir} & {r["id"] for r in dft}
        nir = [r for r in nir if r["id"] in both]
        dft = [r for r in dft if r["id"] in both]
    if not nir:
        return None
    out = {"task": task, "condition": condition, "default": system_stats(dft)}
    out["nirnay"] = system_stats(nir)
    out["nirnay"]["escalated"] = sum(r["escalated"] for r in nir) / len(nir)
    out["nirnay"]["thinking_gave_no_answer"] = sum(
        r["escalated"] and r["escalation_answer"] is None for r in nir
    )

    calib = [r for r in d["calib"] if r["condition"] == condition]
    kept = [r for r in nir if not r["escalated"] and r["raw_confidence"] is not None]
    if calib and kept:
        cal = IsotonicCalibrator().fit(
            [r["raw_confidence"] for r in calib], [r["winner"] == r["label"] for r in calib]
        )
        conf = [cal(r["raw_confidence"]) for r in kept]
        raw = [r["raw_confidence"] for r in kept]
        right = [r["best_guess"] == r["label"] for r in kept]
        out["confidence"] = {
            "dev_items": len(calib),
            "non_escalated": len(kept),
            "accuracy_non_escalated": sum(right) / len(right),
            "mean_calibrated_confidence": statistics.mean(conf),
            "ece_raw": expected_calibration_error(raw, right),
            "ece_calibrated": expected_calibration_error(conf, right),
            "calibrator": dict(zip(cal.xs, cal.ys, strict=True)),
        }

    by_lang = defaultdict(dict)
    for lang in sorted({r["language"] for r in nir}):
        nl = [r for r in nir if r["language"] == lang]
        dl = [r for r in dft if r["language"] == lang]
        by_lang[lang] = {
            "n": len(nl),
            "default_accuracy": acc(dl),
            "nirnay_accuracy": acc(nl),
            "nirnay_escalated": sum(r["escalated"] for r in nl) / len(nl),
        }
        if "confidence" in out:
            kl = [r for r in kept if r["language"] == lang]
            if kl:
                conf = [cal(r["raw_confidence"]) for r in kl]
                right = [r["best_guess"] == r["label"] for r in kl]
                by_lang[lang]["ece_calibrated"] = expected_calibration_error(conf, right)
                by_lang[lang]["mean_confidence"] = statistics.mean(conf)
                by_lang[lang]["accuracy_non_escalated"] = sum(right) / len(right)
    out["by_language"] = dict(by_lang)
    return out


def pct(x):
    return "  –  " if x is None else f"{100 * x:5.1f}%"


def report(s):
    dft, nir = s["default"] or {}, s["nirnay"]
    print(f"\n### {TITLES[s['task']]} — {s['condition']} (test n={nir['n']})")
    print("| | Sarvam default settings | nirnay |\n|---|---|---|")
    print(f"| Accuracy | {pct(dft.get('accuracy'))} | {pct(nir['accuracy'])} |")
    print(f"| No usable answer | {pct(dft.get('unusable'))} | {pct(nir['unusable'])} |")
    cost_d = f"₹{dft['cost_per_1k']:.2f}" if dft else "–"
    print(f"| ₹ per 1,000 | {cost_d} | ₹{nir['cost_per_1k']:.2f} |")
    lat_d = f"{dft['latency_p50']:.0f} / {dft['latency_p95']:.0f}" if dft else "–"
    print(
        f"| Latency p50 / p95 ms | {lat_d} | {nir['latency_p50']:.0f} / {nir['latency_p95']:.0f} |"
    )
    print(f"| Escalated to thinking | – | {pct(nir['escalated'])} |")
    c = s.get("confidence")
    if c:
        print(
            f"confidence: ECE raw {c['ece_raw']:.3f} → calibrated {c['ece_calibrated']:.3f}; "
            f"mean confidence {c['mean_calibrated_confidence']:.2f} vs accuracy "
            f"{pct(c['accuracy_non_escalated'])} on {c['non_escalated']} non-escalated "
            f"(calibrator fit on {c['dev_items']} dev)"
        )
    for lang, v in s["by_language"].items():
        extra = f", ECE {v['ece_calibrated']:.3f}" if "ece_calibrated" in v else ""
        print(
            f"   {lang:9} n={v['n']:3}  default {pct(v['default_accuracy'])}  "
            f"nirnay {pct(v['nirnay_accuracy'])}  escalated {pct(v['nirnay_escalated'])}{extra}"
        )


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = load()
    summary = []
    for task in ORDER:
        if task not in data:
            continue
        for condition in ("described", "names"):
            s = analyse(task, data[task], condition)
            if s:
                summary.append(s)
                report(s)
    (RESULTS / "bench_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
