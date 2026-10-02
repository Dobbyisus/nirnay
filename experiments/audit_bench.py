# ruff: noqa: E501, SIM115, B905
"""Independent audit of every benchmark number, from the raw logs.

Run:  python experiments/audit_bench.py

Deliberately does NOT import bench_analyze. Recomputes accuracy, cost (from tokens x published
prices), latency, escalation; checks pairing, dev/test separation, label provenance; runs
sensitivity analyses for choices that could flatter nirnay; adds paired CIs / McNemar tests.
"""

import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from tasks import choices  # noqa: E402

from nirnay.prompt import label_codes, parse_reply  # noqa: E402

RES = ROOT / "experiments" / "results"
PRIV = ROOT / "data" / "private" / "results"
P_IN, P_CACHED, P_OUT = 29.28, 10.98, 73.20  # ₹ per 1M tokens, sarvam-105b, docs 30 Sep 2026
issues = []


def flag(msg):
    issues.append(msg)
    print("  !! " + msg)


def load_rows(path, default_condition=None):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if default_condition and "condition" not in r:
            r["condition"] = default_condition
        rows.append(r)
    return rows


def successes(rows):
    ok = [r for r in rows if r.get("error") is None]
    c = Counter((r["condition"], r["id"]) for r in ok)
    dups = {k: n for k, n in c.items() if n > 1}
    return ok, dups


def recompute_cost(r):
    unc = r["input_tokens"] - r["cached_input_tokens"]
    return (unc * P_IN + r["cached_input_tokens"] * P_CACHED + r["output_tokens"] * P_OUT) / 1e6


def nearest_rank(vals, q):
    v = sorted(vals)
    return v[max(0, math.ceil(q * len(v)) - 1)]


def nirnay_answer(r, task):
    """Recompute nirnay's final answer from raw replies with the current parser."""
    codes = label_codes(choices(task, r["condition"]))
    order = list(codes)
    votes = Counter(c for c in (parse_reply(x, codes) for x in r.get("raw_replies") or []) if c)
    ranked = sorted(votes, key=lambda c: (-votes[c], order.index(c)))
    winner = codes[ranked[0]] if ranked else None
    share = votes[ranked[0]] / len(r["raw_replies"]) if ranked and r.get("raw_replies") else None
    if r.get("escalated"):
        if r.get("escalation_reply") is not None:
            code = parse_reply(r["escalation_reply"], codes)
            ans = codes[code] if code else None
        else:
            ans = r.get("escalation_answer")
        return (ans or winner), share, winner
    return winner, share, winner


def default_answer(r, task):
    codes = label_codes(choices(task, r["condition"]))
    code = parse_reply(r.get("content"), codes)
    return codes[code] if code else None


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: b = only nirnay right, c = only default right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_ci(pairs, iters=10000, seed=7):
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(iters):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(a - b for a, b in s) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]


def bench_items(task):
    return {
        json.loads(x)["id"]: json.loads(x)
        for x in (ROOT / "data" / "bench" / f"{task}.jsonl").read_text("utf-8").splitlines()
    }


print("=" * 100)
print("A. INTEGRITY: coverage, duplicates, errors, escalation rule, cost recomputation")
DATA = {}
for task in ["top", "massive", "reviews", "tweets"]:
    items = bench_items(task)
    test_ids = {i for i, it in items.items() if it["split"] == "test"}
    dev_ids = {i for i, it in items.items() if it["split"] == "dev"}
    if test_ids & dev_ids:
        flag(f"{task}: dev/test overlap")
    nir_all = load_rows(RES / f"bench_{task}_nirnay.jsonl")
    dft_all = load_rows(RES / f"bench_{task}_default.jsonl")
    nir, nd = successes(nir_all)
    dft, dd = successes(dft_all)
    if nd or dd:
        flag(f"{task}: duplicate successful rows nirnay={len(nd)} default={len(dd)}")
    for cond in ["names", "described"]:
        n_ids = {r["id"] for r in nir if r["condition"] == cond and r["split"] == "test"}
        d_ids = {r["id"] for r in dft if r["condition"] == cond}
        if n_ids != test_ids or d_ids != test_ids:
            flag(
                f"{task}/{cond}: coverage nirnay {len(n_ids)} default {len(d_ids)} of {len(test_ids)}"
            )
    # labels in result rows must match the bench file
    bad = sum(r["label"] != items[r["id"]]["label"] for r in nir + dft)
    if bad:
        flag(f"{task}: {bad} rows with label mismatch vs bench file")
    # escalation rule
    viol = sum(
        (r["escalated"] != (r["raw_confidence"] is None or r["raw_confidence"] < 0.8)) for r in nir
    )
    if viol:
        flag(f"{task}: {viol} rows break the escalate-below-0.8 rule")
    # cost recomputation
    cmax = max(abs(recompute_cost(r) - r["cost_inr"]) for r in nir + dft)
    print(
        f"{task:8} test={len(test_ids)} dev={len(dev_ids)} nirnay rows={len(nir)} default rows={len(dft)} "
        f"| max |recorded cost - recomputed| = {cmax:.2e}"
    )
    DATA[task] = {"nir": nir, "dft": dft, "items": items}

# support (private)
s_items = {
    json.loads(x)["id"]: json.loads(x)
    for x in (ROOT / "data" / "private" / "support_v0.jsonl").read_text("utf-8").splitlines()
}
s_nir = (
    successes(load_rows(PRIV / "support_v0_nirnay_live.jsonl", "described"))[0]
    + successes(load_rows(PRIV / "support_v0_nirnay_live_names.jsonl", "names"))[0]
)
s_dft, sdd = successes(load_rows(PRIV / "support_v0_default.jsonl"))
s_votes, _ = successes(load_rows(PRIV / "support_v0_votes.jsonl"))
s_test = {i for i, it in s_items.items() if it["split"] == "test"}
for cond in ["names", "described"]:
    n_ids = {r["id"] for r in s_nir if r["condition"] == cond}
    d_ids = {r["id"] for r in s_dft if r["condition"] == cond and r["split"] == "test"}
    if n_ids != s_test or d_ids != s_test:
        flag(f"support/{cond}: coverage nirnay {len(n_ids)} default {len(d_ids)} of {len(s_test)}")
cmax = max(abs(recompute_cost(r) - r["cost_inr"]) for r in s_nir + s_dft)
print(
    f"support  test={len(s_test)} nirnay rows={len(s_nir)} default rows={len(s_dft)} | max cost diff {cmax:.2e}"
)
DATA["support"] = {
    "nir": s_nir,
    "dft": [r for r in s_dft if r["split"] == "test"],
    "items": s_items,
}

print("\n" + "=" * 100)
print("B. LABEL PROVENANCE: bench labels vs the original source files (random 40 per task)")
rng = random.Random(1)
raw = ROOT / "data" / "raw"
# TOP
top_src = {}
for split, f in [("test", "test.tsv"), ("dev", "validation.tsv")]:
    with open(raw / "top" / f, encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh, delimiter="\t")):
            top_src[f"{split}{i}"] = r["domain"]
mis = sum(
    top_src[it["pair"]] != it["label"] for it in rng.sample(list(DATA["top"]["items"].values()), 40)
)
print(f"top: {mis}/40 mismatches")
# MASSIVE
mas = {}
for loc in ["en-US", "hi-IN", "bn-BD", "ta-IN"]:
    with open(raw / "massive" / "1.1" / "data" / f"{loc}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            mas[(loc, d["id"])] = (d["scenario"], d["utt"], d["partition"])
locmap = {"English": "en-US", "Hindi": "hi-IN", "Bengali": "bn-BD", "Tamil": "ta-IN"}
smp = rng.sample(list(DATA["massive"]["items"].values()), 40)
mis = sum(mas[(locmap[it["language"]], it["pair"])][0] != it["label"] for it in smp)
part = sum(
    mas[(locmap[it["language"]], it["pair"])][2] != ("test" if it["split"] == "test" else "dev")
    for it in smp
)
print(f"massive: {mis}/40 label mismatches, {part}/40 partition mismatches")
# reviews
revsrc = {}
for split, part in [("test", "test"), ("dev", "validation")]:
    for code, lang in [("hi", "Hindi"), ("bn", "Bengali"), ("ta", "Tamil")]:
        rows = [
            json.loads(x)
            for x in open(raw / "indicsentiment" / f"{part}_{code}.json", encoding="utf-8")
            if x.strip()
        ]
        for i, r in enumerate(rows):
            revsrc[(lang, f"{split}{i}")] = (r["LABEL"] or "").lower()
            revsrc[("English", f"{split}{i}")] = (r["LABEL"] or "").lower()
mis = sum(
    revsrc[(it["language"], it["pair"])] != it["label"]
    for it in rng.sample(list(DATA["reviews"]["items"].values()), 40)
)
print(f"reviews: {mis}/40 mismatches")
# tweets
lab = {
    r["Uid"]: r["Sentiment"]
    for r in csv.DictReader(
        open(
            raw / "sentimix" / "Semeval_2020_task9_data" / "Hinglish" / "Hinglish_test_labels.txt",
            encoding="utf-8",
        )
    )
}
tw_test = [it for it in DATA["tweets"]["items"].values() if it["split"] == "test"]
mis = sum(lab[it["pair"].replace("test", "")] != it["label"] for it in tw_test)
print(f"tweets: {mis}/{len(tw_test)} test-label mismatches vs labels file")
dist = {
    t: Counter(it["label"] for it in DATA[t]["items"].values() if it["split"] == "test")
    for t in ["top", "massive", "reviews", "tweets"]
}
print("test label balance:", {t: (min(c.values()), max(c.values())) for t, c in dist.items()})

print("\n" + "=" * 100)
print("C. RECOMPUTED HEADLINE METRICS (paired, test sets, current parser for BOTH systems)")
SUMMARY = {}
for task in ["support", "top", "massive", "reviews", "tweets"]:
    d = DATA[task]
    for cond in ["described", "names"]:
        nir = {
            r["id"]: r
            for r in d["nir"]
            if r["condition"] == cond and r.get("split", "test") == "test"
        }
        dft = {r["id"]: r for r in d["dft"] if r["condition"] == cond}
        ids = sorted(set(nir) & set(dft))
        n = len(ids)
        if task == "support":
            nans = {
                i: nir[i]["best_guess"] for i in ids
            }  # support live rows: no raw thinking reply saved
        else:
            nans = {i: nirnay_answer(nir[i], task)[0] for i in ids}
        dans = {i: default_answer(dft[i], task) for i in ids}
        lab = {i: d["items"][i]["label"] for i in ids}
        n_ok = [nans[i] == lab[i] for i in ids]
        d_ok = [dans[i] == lab[i] for i in ids]
        d_unusable = sum(dans[i] is None for i in ids)
        n_unusable = sum(nans[i] is None for i in ids)
        d_answered = [i for i in ids if dans[i] is not None]
        b = sum(a and not c for a, c in zip(n_ok, d_ok))
        c_ = sum(c and not a for a, c in zip(n_ok, d_ok))
        lo, hi = paired_ci(list(zip(map(int, n_ok), map(int, d_ok))))
        nc = [nir[i]["cost_inr"] for i in ids]
        dc = [dft[i]["cost_inr"] for i in ids]
        nl = [nir[i]["latency_ms"] for i in ids]
        dl = [dft[i]["latency_ms"] for i in ids]
        esc = sum(nir[i]["escalated"] for i in ids)
        recorded_n = sum(nir[i]["best_guess"] == lab[i] for i in ids) / n
        SUMMARY[(task, cond)] = dict(
            n=n,
            n_acc=sum(n_ok) / n,
            d_acc=sum(d_ok) / n,
            b=b,
            c=c_,
            ci=(lo, hi),
            p=mcnemar_exact(b, c_),
            d_unusable=d_unusable,
            n_unusable=n_unusable,
            d_acc_answered=sum(dans[i] == lab[i] for i in d_answered) / len(d_answered),
            n_cost=1000 * sum(nc) / n,
            d_cost=1000 * sum(dc) / n,
            n_p50=nearest_rank(nl, 0.5),
            d_p50=nearest_rank(dl, 0.5),
            n_p95=nearest_rank(nl, 0.95),
            d_p95=nearest_rank(dl, 0.95),
            esc=esc / n,
            n_ok=n_ok,
            d_ok=d_ok,
            recorded_n=recorded_n,
        )
        s = SUMMARY[(task, cond)]
        print(
            f"{task:8}{cond:10} n={n:3} default {100 * s['d_acc']:5.1f}% nirnay {100 * s['n_acc']:5.1f}% "
            f"diff {100 * (s['n_acc'] - s['d_acc']):+5.1f} pts  95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]  "
            f"McNemar p={s['p']:.3f} (only-nirnay-right {b}, only-default-right {c_}) | "
            f"₹/1k {s['d_cost']:.2f} -> {s['n_cost']:.2f} ({100 * s['n_cost'] / s['d_cost']:.0f}%) | "
            f"p50 {s['d_p50']:.0f}->{s['n_p50']:.0f} ms, p95 {s['d_p95']:.0f}->{s['n_p95']:.0f} | esc {100 * s['esc']:.1f}% | "
            f"default unusable {d_unusable}, nirnay unusable {n_unusable}"
        )
        if abs(recorded_n - s["n_acc"]) > 1e-9:
            print(
                f"          note: nirnay accuracy from recorded answers {100 * recorded_n:.1f}% vs recomputed {100 * s['n_acc']:.1f}%"
            )

print("\n" + "=" * 100)
print("D. POOLED (described) and sensitivity checks")
pairs, nc_tot, dc_tot, N = [], 0, 0, 0
for task in ["support", "top", "massive", "reviews", "tweets"]:
    s = SUMMARY[(task, "described")]
    pairs += list(zip(map(int, s["n_ok"]), map(int, s["d_ok"])))
    nc_tot += s["n_cost"] * s["n"]
    dc_tot += s["d_cost"] * s["n"]
    N += s["n"]
na = sum(a for a, _ in pairs) / N
da = sum(b for _, b in pairs) / N
b = sum(a and not c for a, c in pairs)
c_ = sum(c and not a for a, c in pairs)
lo, hi = paired_ci(pairs)
print(
    f"pooled described n={N}: default {100 * da:.1f}% nirnay {100 * na:.1f}% diff {100 * (na - da):+.1f} "
    f"CI [{100 * lo:+.1f}, {100 * hi:+.1f}] McNemar p={mcnemar_exact(b, c_):.4f} | ₹/1k {dc_tot / N:.2f} -> {nc_tot / N:.2f} ({100 * nc_tot / dc_tot:.0f}%)"
)
unw_n = (
    sum(
        SUMMARY[(t, "described")]["n_acc"]
        for t in ["support", "top", "massive", "reviews", "tweets"]
    )
    / 5
)
unw_d = (
    sum(
        SUMMARY[(t, "described")]["d_acc"]
        for t in ["support", "top", "massive", "reviews", "tweets"]
    )
    / 5
)
print(
    f"unweighted mean of 5 tasks (described): default {100 * unw_d:.1f}% nirnay {100 * unw_n:.1f}%"
)

print("\nD1. If Sarvam default had a fallback (accuracy on the decisions it DID answer):")
for task in ["support", "top", "massive", "reviews", "tweets"]:
    s = SUMMARY[(task, "described")]
    print(
        f"   {task:8} default answered-only {100 * s['d_acc_answered']:.1f}% (vs {100 * s['d_acc']:.1f}% counting no-answer as wrong) | nirnay {100 * s['n_acc']:.1f}%"
    )

print("\nD2. Parser effect on the DEFAULT (runtime parser vs current parser):")
for task in ["top", "massive", "reviews", "tweets"]:
    for cond in ["described", "names"]:
        rows = [r for r in DATA[task]["dft"] if r["condition"] == cond]
        rt = sum(r["best_guess"] == r["label"] for r in rows) / len(rows)
        cur = sum(default_answer(r, task) == r["label"] for r in rows) / len(rows)
        print(
            f"   {task:8}{cond:10} runtime-parsed {100 * rt:5.1f}% -> current parser {100 * cur:5.1f}%"
        )

print("\nD3. Rerun selection effect (152 nirnay decisions rerun after the parser fix):")
for task in ["top", "massive"]:
    arch = [
        json.loads(x)
        for x in open(
            RES / "archive" / f"bench_{task}_nirnay_before_parser_fix.jsonl", encoding="utf-8"
        )
    ]
    rem = {
        (r["condition"], r["id"]): r
        for r in arch
        if r.get("error") is None
        and (r["invalid_samples"] > 0 or (r["escalated"] and r["escalation_answer"] is None))
    }
    for cond in ["described", "names"]:
        cur = {
            r["id"]: r for r in DATA[task]["nir"] if r["condition"] == cond and r["split"] == "test"
        }
        ids = sorted(cur)
        lab = {i: DATA[task]["items"][i]["label"] for i in ids}
        acc_cur = sum(nirnay_answer(cur[i], task)[0] == lab[i] for i in ids) / len(ids)
        swapped = 0
        ok = 0
        for i in ids:
            r = rem.get((cond, i))
            if r is not None:
                swapped += 1
                ans, _, winner = nirnay_answer({**r, "escalation_reply": None}, task)
                ok += ans == lab[i]
            else:
                ok += nirnay_answer(cur[i], task)[0] == lab[i]
        print(
            f"   {task:8}{cond:10} test items replaced by reruns: {swapped:3} | accuracy with reruns {100 * acc_cur:.1f}% "
            f"vs with ORIGINAL (pre-fix) results {100 * ok / len(ids):.1f}%"
        )

print(
    "\nD4. Escalation threshold: 0.8 was picked from the SUPPORT test-set replay; other tasks are out-of-sample."
)
print("D5. Calibration fit on dev only; dev/test id overlap checked in A.")

print("\n" + "=" * 100)
print("E. CONFIDENCE (ECE on non-escalated test decisions; isotonic fit on dev)")
from nirnay.calibration import IsotonicCalibrator, expected_calibration_error  # noqa: E402

for task in ["support", "top", "massive", "reviews", "tweets"]:
    for cond in ["described", "names"]:
        if task == "support":
            dev = [r for r in s_votes if r["condition"] == cond and r["split"] == "dev"]
            devx = [(r["raw_confidence"], r["best_guess"] == r["label"]) for r in dev]
        else:
            dev = [r for r in DATA[task]["nir"] if r["condition"] == cond and r["split"] == "dev"]
            devx = [
                (r["raw_confidence"], r["vote_winner"] == r["label"])
                for r in dev
                if r["raw_confidence"] is not None
            ]
        test = [
            r
            for r in DATA[task]["nir"]
            if r["condition"] == cond
            and r.get("split", "test") == "test"
            and not r["escalated"]
            and r["raw_confidence"] is not None
        ]
        cal = IsotonicCalibrator().fit([x for x, _ in devx], [y for _, y in devx])
        raw_c = [r["raw_confidence"] for r in test]
        ok = [r["best_guess"] == r["label"] for r in test]
        allt = [
            r
            for r in DATA[task]["nir"]
            if r["condition"] == cond and r.get("split", "test") == "test"
        ]
        print(
            f"   {task:8}{cond:10} dev={len(devx):3} non-escalated test={len(test):3}/{len(allt)} "
            f"acc {100 * sum(ok) / len(ok):5.1f}% mean raw conf {100 * sum(raw_c) / len(raw_c):5.1f}% "
            f"ECE raw {expected_calibration_error(raw_c, ok):.3f} -> calibrated "
            f"{expected_calibration_error([cal(x) for x in raw_c], ok):.3f}"
        )

print("\n" + "=" * 100)
print("ISSUES FLAGGED:", issues if issues else "none")
