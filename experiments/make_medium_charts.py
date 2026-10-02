# ruff: noqa: E501  (chart layout: long coordinate lines kept on one line for readability)
"""Article / README charts in a "card" style: kicker, title, subtitle, legend, paired bars per
row with coloured value labels, a delta pill per row, an optional highlighted row, footnote.

Reads experiments/results/bench_summary.json (run bench_analyze.py first). Writes 1200x800 PNGs
(plus @2x) and captions to deliverables/charts/, and a zip next to them.

Run:  python experiments/make_medium_charts.py
"""

import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "experiments" / "results" / "bench_summary.json"
OUT = ROOT / "deliverables" / "charts"
CREDIT = "nirnay · open-source decision layer for the Sarvam API"  # bottom-left credit line

# Validated with the dataviz palette checker (light surface): CVD ΔE 16.0, all checks pass.
PINK, PURPLE, GREY = "#D9508A", "#7657E6", "#A8A49C"
INK, INK2, MUTED, RULE = "#16161A", "#5B5A57", "#8A8883", "#ECEAE6"
HIGHLIGHT, GREEN_BG, GREEN_EDGE, GREEN_TXT = "#FCEAF1", "#EAF7EC", "#7CC48A", "#1E7B35"
PILL_BG, PILL_EDGE = "#F4F3F1", "#D9D6D0"
NIRNAY, DEFAULT = "nirnay", "Sarvam default settings"

# Exact McNemar p-values from experiments/audit_bench.py (paired, test sets, with descriptions).
SIGNIFICANT_DESCRIBED = {"massive", "reviews", "all"}  # p = 0.016, 0.008, 0.002; others p ≥ 0.73

TASKS = {
    "support": ("Support routing", "Hindi, Hinglish · 11 categories"),
    "top": ("Voice-assistant domain", "English vs Hinglish, same requests · 8"),
    "massive": ("Voice-assistant scenario", "English, Hindi, Bengali, Tamil · 18"),
    "reviews": ("Review sentiment", "English, Hindi, Bengali, Tamil · 2"),
    "tweets": ("Tweet sentiment", "real code-mixed Hinglish tweets · 3"),
}
ORDER = list(TASKS)

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "text.color": INK})


def spaced(text):
    return " ".join(text)  # thin spaces imitate letter-spaced small caps


def pill(fig, x, y, text, good):
    text = "−" + text[1:] if text.startswith("-") else text  # typographic minus
    bg, edge, color = (GREEN_BG, GREEN_EDGE, GREEN_TXT) if good else (PILL_BG, PILL_EDGE, INK2)
    w, h = 0.075, 0.042
    fig.patches.append(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                      transform=fig.transFigure, facecolor=bg, edgecolor=edge, linewidth=1))  # fmt: skip
    fig.text(x, y, text, ha="center", va="center", fontsize=11, color=color, fontweight="bold")


def card(spec, name):
    """spec: kicker, title, subtitle, legend [(label, color)], right_header, rows, xmax, fmt,
    footnote. Row: label, sub, a, b, pill, good, highlight."""
    fig = plt.figure(figsize=(12, 8), dpi=100)
    fig.patch.set_facecolor("white")
    fig.patches.append(FancyBboxPatch((0.012, 0.015), 0.976, 0.97, boxstyle="round,pad=0,rounding_size=0.015",
                                      transform=fig.transFigure, facecolor="white", edgecolor=RULE, linewidth=1.2))  # fmt: skip
    L = 0.055
    fig.add_artist(plt.Line2D([L, L + 0.028], [0.927, 0.927], color=PINK, linewidth=3,
                              transform=fig.transFigure, solid_capstyle="round"))  # fmt: skip
    fig.text(
        L + 0.038,
        0.927,
        spaced(spec["kicker"].upper()),
        fontsize=10,
        color=PINK,
        va="center",
        fontweight="bold",
    )
    fig.text(L, 0.868, spec["title"], fontsize=23, fontweight="bold", va="center", color=INK)
    fig.text(L, 0.818, spec["subtitle"], fontsize=12.5, va="center", color=INK2)
    x = L
    for label, color in spec["legend"]:
        fig.patches.append(FancyBboxPatch((x, 0.759), 0.012, 0.018, boxstyle="round,pad=0,rounding_size=0.003",
                                          transform=fig.transFigure, facecolor=color, edgecolor="none"))  # fmt: skip
        t = fig.text(x + 0.018, 0.768, label, fontsize=11.5, va="center", color=INK)
        x += 0.018 + 0.0072 * len(label) + 0.04
    _ = t
    fig.text(
        0.945,
        0.768,
        spaced(spec["right_header"].upper()),
        fontsize=9.5,
        ha="right",
        va="center",
        color=MUTED,
    )

    rows = spec["rows"]
    top, bottom = 0.735, 0.17
    rh = (top - bottom) / len(rows)
    bx0, bx1 = 0.35, 0.80
    for i, r in enumerate(rows):
        yc = top - (i + 0.5) * rh
        if r.get("highlight"):
            fig.patches.append(FancyBboxPatch((0.035, yc - rh / 2 + 0.004), 0.93, rh - 0.008,
                                              boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                              facecolor=HIGHLIGHT, edgecolor="#F3C6D8", linewidth=1))  # fmt: skip
        elif i > 0:
            fig.add_artist(plt.Line2D([0.045, 0.955], [yc + rh / 2, yc + rh / 2], color=RULE, linewidth=1,
                                      transform=fig.transFigure))  # fmt: skip
        fig.text(
            L, yc + rh * 0.13, r["label"], fontsize=13.5, fontweight="bold", va="center", color=INK
        )
        fig.text(L, yc - rh * 0.20, r["sub"], fontsize=10.5, va="center", color=MUTED)
        bh = min(0.022, rh * 0.25)
        for j, (val, color) in enumerate(
            [(r["a"], spec["legend"][0][1]), (r["b"], spec["legend"][1][1])]
        ):
            y = yc + (0.55 if j == 0 else -0.55) * bh * 1.15
            w = (bx1 - bx0) * max(val, 0) / spec["xmax"]
            if w > 0:
                fig.patches.append(FancyBboxPatch((bx0, y - bh / 2), w, bh, boxstyle="round,pad=0,rounding_size=0.003",
                                                  transform=fig.transFigure, facecolor=color, edgecolor="none"))  # fmt: skip
            fig.text(
                bx0 + w + 0.006,
                y,
                spec["fmt"](val),
                fontsize=10.5,
                va="center",
                color=color,
                fontweight="bold",
            )
        pill(fig, 0.905, yc, r["pill"], r.get("good", False))
    fig.text(
        L,
        0.105,
        spec["footnote"],
        fontsize=10.5,
        color=INK2,
        va="center",
        style="italic",
        wrap=True,
    )
    fig.text(L, 0.048, CREDIT, fontsize=10, color=MUTED, va="center", fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=100)
    fig.savefig(OUT / f"{name}@2x.png", dpi=200)
    plt.close(fig)
    return name


def main():
    S = {(s["task"], s["condition"]): s for s in json.loads(SUMMARY.read_text(encoding="utf-8"))}
    D = {t: S[(t, "described")] for t in ORDER}
    N = {t: S[(t, "names")] for t in ORDER}
    n_all = sum(D[t]["nirnay"]["n"] for t in ORDER)

    def pooled(key, system):
        return sum(D[t][system][key] * D[t][system]["n"] for t in ORDER) / n_all

    pct = lambda v: f"{v:.1f}%"  # noqa: E731
    captions = {}

    def task_rows(src, a_fn, b_fn, pill_fn, good_fn, highlight_fn=lambda t: False):
        return [{"label": TASKS[t][0], "sub": f"{TASKS[t][1]} · n={src[t]['nirnay']['n']}", "a": a_fn(t),
                 "b": b_fn(t), "pill": pill_fn(t), "good": good_fn(t), "highlight": highlight_fn(t)}
                for t in ORDER]  # fmt: skip

    # 1. Accuracy with descriptions
    rows = task_rows(D, lambda t: 100 * D[t]["nirnay"]["accuracy"], lambda t: 100 * D[t]["default"]["accuracy"],
                     lambda t: f"{100 * (D[t]['nirnay']['accuracy'] - D[t]['default']['accuracy']):+.1f}",
                     lambda t: t in SIGNIFICANT_DESCRIBED, lambda t: False)  # fmt: skip
    a, b = 100 * pooled("accuracy", "nirnay"), 100 * pooled("accuracy", "default")
    rows.append({"label": "All decisions", "sub": f"5 tasks, 6 language varieties · n={n_all}", "a": a, "b": b,
                 "pill": f"{a - b:+.1f}", "good": True, "highlight": True})  # fmt: skip
    captions[card({
        "kicker": "nirnay · accuracy", "title": "Similar accuracy to Sarvam's default settings",
        "subtitle": "Share of test decisions answered correctly, with label descriptions. Same model: sarvam-105b.",
        "legend": [(NIRNAY, PINK), (DEFAULT, PURPLE)], "right_header": "nirnay − default (pts)",
        "rows": rows, "xmax": 100, "fmt": pct,
        "footnote": "Green = statistically significant (paired, p < 0.05). Decisions the default left unanswered count as wrong.",
    }, "01_accuracy_with_descriptions")] = (
        "Accuracy by task with label descriptions: nirnay vs Sarvam default settings (sarvam-105b).",
        "nirnay matches Sarvam's default accuracy on every task; only voice scenario (+2.8) and review "
        "sentiment (+1.7) are statistically significant. Overall 91.2% vs 89.7% on 1,616 decisions.")  # fmt: skip

    # 2. Accuracy, names only
    rows = task_rows(N, lambda t: 100 * N[t]["nirnay"]["accuracy"], lambda t: 100 * N[t]["default"]["accuracy"],
                     lambda t: f"{100 * (N[t]['nirnay']['accuracy'] - N[t]['default']['accuracy']):+.1f}",
                     lambda t: False)  # fmt: skip
    captions[card({
        "kicker": "nirnay · accuracy", "title": "Without label descriptions, it's a draw",
        "subtitle": "Accuracy with category names only (no descriptions). None of these differences is significant.",
        "legend": [(NIRNAY, PINK), (DEFAULT, PURPLE)], "right_header": "nirnay − default (pts)",
        "rows": rows, "xmax": 100, "fmt": pct,
        "footnote": "With vague names, nirnay's votes can agree on the wrong answer, so it escalates more and saves less.",
    }, "02_accuracy_names_only")] = (
        "Accuracy by task with category names only.",
        "Without descriptions nirnay is within a few points of the default either way (support −2.1, tweets −2.0; "
        "not significant). Descriptions are what make nirnay work well.")  # fmt: skip

    # 3. Cost
    rows = task_rows(D, lambda t: D[t]["nirnay"]["cost_per_1k"], lambda t: D[t]["default"]["cost_per_1k"],
                     lambda t: f"{100 * D[t]['nirnay']['cost_per_1k'] / D[t]['default']['cost_per_1k']:.0f}%",
                     lambda t: True)  # fmt: skip
    a, b = pooled("cost_per_1k", "nirnay"), pooled("cost_per_1k", "default")
    rows.append({"label": "All decisions", "sub": f"weighted by task size · n={n_all}", "a": a, "b": b,
                 "pill": f"{100 * a / b:.0f}%", "good": True, "highlight": True})  # fmt: skip
    captions[
        card(
            {
                "kicker": "nirnay · cost",
                "title": "About 3.5× cheaper per decision",
                "subtitle": "Rupees per 1,000 decisions at Sarvam's published prices (₹29.28 in / ₹10.98 cached / ₹73.20 out per 1M).",
                "legend": [(NIRNAY, PINK), (DEFAULT, PURPLE)],
                "right_header": "nirnay ÷ default",
                "rows": rows,
                "xmax": 62,
                "fmt": lambda v: f"₹{v:.2f}",
                "footnote": "nirnay's cost includes the thinking call for every escalated decision.",
            },
            "03_cost_per_1000",
        )
    ] = (
        "Cost in rupees per 1,000 decisions, with label descriptions.",
        "nirnay costs 14–38% of Sarvam's default settings per task, 28% overall (₹11.97 vs ₹43.51 per 1,000).",
    )

    # 4. Typical latency
    rows = task_rows(D, lambda t: D[t]["nirnay"]["latency_p50"] / 1000, lambda t: D[t]["default"]["latency_p50"] / 1000,
                     lambda t: f"{D[t]['default']['latency_p50'] / D[t]['nirnay']['latency_p50']:.0f}×",
                     lambda t: True)  # fmt: skip
    captions[
        card(
            {
                "kicker": "nirnay · speed",
                "title": "12–21× faster for a typical decision",
                "subtitle": "Median response time per decision, in seconds, with label descriptions.",
                "legend": [(NIRNAY, PINK), (DEFAULT, PURPLE)],
                "right_header": "times faster",
                "rows": rows,
                "xmax": 4.0,
                "fmt": lambda v: f"{v:.2f} s",
                "footnote": "Measured from the client in India against Sarvam's hosted API; rate-limit pauses excluded.",
            },
            "04_latency_typical",
        )
    ] = (
        "Median (p50) response time per decision.",
        "nirnay answers a typical decision in about 0.18 s vs 2.2–3.5 s for Sarvam's default settings.",
    )

    # 5. Tail latency
    rows = task_rows(D, lambda t: D[t]["nirnay"]["latency_p95"] / 1000, lambda t: D[t]["default"]["latency_p95"] / 1000,
                     lambda t: f"{D[t]['default']['latency_p95'] / D[t]['nirnay']['latency_p95']:.1f}×",
                     lambda t: D[t]["default"]["latency_p95"] / D[t]["nirnay"]["latency_p95"] >= 2)  # fmt: skip
    captions[
        card(
            {
                "kicker": "nirnay · speed",
                "title": "The slowest decisions: still faster, but less so",
                "subtitle": "95th-percentile response time, in seconds. Escalated decisions wait for Sarvam's thinking mode.",
                "legend": [(NIRNAY, PINK), (DEFAULT, PURPLE)],
                "right_header": "times faster",
                "rows": rows,
                "xmax": 12,
                "fmt": lambda v: f"{v:.1f} s",
                "footnote": "Tasks with more escalations (voice scenario 16.5%, tweets 20%) have the slowest tails.",
            },
            "05_latency_slowest_5pct",
        )
    ] = (
        "95th-percentile (p95) response time per decision.",
        "On the slowest 5% of decisions nirnay is 1.3–12× faster; escalated decisions take as long as a thinking call.",
    )

    # 6. Failed answers
    rows = task_rows(D, lambda t: 100 * D[t]["nirnay"]["unusable"], lambda t: 100 * D[t]["default"]["unusable"],
                     lambda t: f"{round(D[t]['default']['unusable'] * D[t]['default']['n'])} vs 0",
                     lambda t: True)  # fmt: skip
    captions[card({
        "kicker": "nirnay · reliability", "title": "nirnay never failed to answer",
        "subtitle": "Share of decisions with no usable answer, with label descriptions.",
        "legend": [(NIRNAY, PINK), (DEFAULT, PURPLE)], "right_header": "failed: default vs nirnay",
        "rows": rows, "xmax": 6.5, "fmt": pct,
        "footnote": "The default's failures were mostly thinking until it hit the 2,048-token limit; nirnay falls back to its vote winner.",
    }, "06_failed_answers")] = (
        "Decisions with no usable answer.",
        "Sarvam's default settings gave no usable answer on 1.2–5.4% of decisions (11.3% on voice scenario without "
        "descriptions); nirnay on 0%.")  # fmt: skip

    # 7. Honesty chart: default accuracy on the decisions it answered
    def answered(t):
        d = D[t]["default"]
        return 100 * d["accuracy"] / (1 - d["unusable"])  # unanswered are always wrong

    rows = task_rows(D, lambda t: 100 * D[t]["nirnay"]["accuracy"], answered,
                     lambda t: f"{100 * D[t]['nirnay']['accuracy'] - answered(t):+.1f}", lambda t: False)  # fmt: skip
    captions[card({
        "kicker": "nirnay · the fine print", "title": "When the default does answer, it's a bit more accurate",
        "subtitle": "nirnay (all decisions) vs Sarvam default settings on only the decisions it answered.",
        "legend": [(NIRNAY, PINK), (DEFAULT + " (answered only)", PURPLE)], "right_header": "nirnay − default (pts)",
        "rows": rows, "xmax": 100, "fmt": pct,
        "footnote": "So nirnay's accuracy edge mostly comes from the default's failed answers; its clear wins are cost, speed and reliability.",
    }, "07_default_answered_only")] = (
        "nirnay accuracy vs Sarvam default accuracy on only the decisions the default answered.",
        "Transparency chart: excluding its failed answers, the default is 0.3–3.1 points more accurate on four of five "
        "tasks. This view flatters the default somewhat, since its unanswered items were probably harder.")  # fmt: skip

    # 8. Label descriptions (nirnay)
    rows = task_rows(D, lambda t: 100 * D[t]["nirnay"]["accuracy"], lambda t: 100 * N[t]["nirnay"]["accuracy"],
                     lambda t: f"{100 * (D[t]['nirnay']['accuracy'] - N[t]['nirnay']['accuracy']):+.1f}",
                     lambda t: D[t]["nirnay"]["accuracy"] - N[t]["nirnay"]["accuracy"] > 0.05,
                     lambda t: D[t]["nirnay"]["accuracy"] - N[t]["nirnay"]["accuracy"] > 0.15)  # fmt: skip
    captions[card({
        "kicker": "nirnay · label descriptions", "title": "One line per category: up to +17 points",
        "subtitle": "nirnay's accuracy with a short description per category vs category names only.",
        "legend": [("with descriptions", PINK), ("names only", GREY)], "right_header": "gain (pts)",
        "rows": rows, "xmax": 100, "fmt": pct,
        "footnote": "Descriptions matter when names are ambiguous ('cancel' meaning cancellation fees; 'qa', 'iot').",
    }, "08_label_descriptions")] = (
        "Effect of label descriptions on nirnay's accuracy.",
        "Descriptions add 17.4 points on support routing and 10.7 on voice scenarios, and about nothing where the "
        "category names are self-explanatory.")  # fmt: skip

    # 9. Languages
    lang_rows = []
    for t, langs in [
        ("massive", ["Tamil", "Bengali", "Hindi", "English"]),
        ("reviews", ["Tamil", "Bengali", "Hindi", "English"]),
    ]:
        for lang in langs:
            v = D[t]["by_language"][lang]
            lang_rows.append({"label": f"{TASKS[t][0].replace('Voice-assistant ', '').capitalize()} · {lang}",
                              "sub": f"{TASKS[t][0]} · n={v['n']}", "a": 100 * v["nirnay_accuracy"],
                              "b": 100 * v["default_accuracy"],
                              "pill": f"{100 * (v['nirnay_accuracy'] - v['default_accuracy']):+.1f}", "good": False})  # fmt: skip
    captions[
        card(
            {
                "kicker": "nirnay · languages",
                "title": "Biggest gains in Tamil and Bengali",
                "subtitle": "Accuracy by language on the same requests and reviews in four languages, with descriptions.",
                "legend": [(NIRNAY, PINK), (DEFAULT, PURPLE)],
                "right_header": "nirnay − default (pts)",
                "rows": lang_rows,
                "xmax": 100,
                "fmt": pct,
                "footnote": "About 120 items per language: gaps under ~4 points are within noise. Part of the default's loss is failed answers.",
            },
            "09_accuracy_by_language",
        )
    ] = (
        "Accuracy by language on parallel data (MASSIVE scenarios and IndicSentiment reviews).",
        "nirnay gains +6.3 (Tamil) and +5.6 (Bengali) on voice scenarios; English and Hindi are within a point.",
    )

    # 10. Code-mixing
    cm = [
        ("top", "English"),
        ("top", "Hinglish"),
        ("support", "Hindi"),
        ("support", "Hinglish"),
        ("tweets", "Hinglish"),
    ]
    cm_rows = []
    for t, lang in cm:
        v = D[t]["by_language"][lang]
        cm_rows.append({"label": f"{lang}", "sub": f"{TASKS[t][0]} · n={v['n']}", "a": 100 * v["nirnay_accuracy"],
                        "b": 100 * v["default_accuracy"],
                        "pill": f"{100 * (v['nirnay_accuracy'] - v['default_accuracy']):+.1f}", "good": False})  # fmt: skip
    captions[card({
        "kicker": "nirnay · hinglish", "title": "Hinglish costs both systems a few points",
        "subtitle": "Accuracy on English, Hindi and code-mixed Hinglish text, with descriptions.",
        "legend": [(NIRNAY, PINK), (DEFAULT, PURPLE)], "right_header": "nirnay − default (pts)",
        "rows": cm_rows, "xmax": 100, "fmt": pct,
        "footnote": "Voice domain uses the same requests in English and Hinglish: both systems drop about 3 points on Hinglish.",
    }, "10_hinglish")] = (
        "Accuracy on English vs code-mixed Hinglish.",
        "On identical requests, accuracy falls from 97.5% (English) to 95.0% (nirnay) / 94.2% (default) in Hinglish. "
        "Real Hinglish tweets are hardest for both (~66%).")  # fmt: skip

    # 11. Confidence honesty
    rows = task_rows(D, lambda t: 100 * D[t]["confidence"]["mean_calibrated_confidence"],
                     lambda t: 100 * D[t]["confidence"]["accuracy_non_escalated"],
                     lambda t: f"{100 * (D[t]['confidence']['accuracy_non_escalated'] - D[t]['confidence']['mean_calibrated_confidence']):+.1f}",
                     lambda t: abs(D[t]["confidence"]["accuracy_non_escalated"] - D[t]["confidence"]["mean_calibrated_confidence"]) <= 0.025)  # fmt: skip
    for r, t in zip(rows, ORDER, strict=True):
        r["sub"] = f"{D[t]['confidence']['non_escalated']} decisions answered without escalating"
    captions[card({
        "kicker": "nirnay · confidence", "title": "How honest is nirnay's confidence?",
        "subtitle": "Average confidence nirnay reported vs how often it was actually right (non-escalated decisions).",
        "legend": [("confidence reported", PINK), ("actually right", GREY)], "right_header": "right − claimed (pts)",
        "rows": rows, "xmax": 100, "fmt": pct,
        "footnote": "Positive = right more often than it claimed (conservative). Green = within 2.5 points. Calibrated on separate dev data.",
    }, "11_confidence_honesty")] = (
        "Is nirnay's confidence honest? Reported confidence vs actual accuracy.",
        "Reported confidence is within 2.5 points of actual accuracy on support, voice domain and reviews, and "
        "conservative on voice scenario and tweets (it is right more often than it claims).")  # fmt: skip

    # 12. Escalation rate
    rows = task_rows(D, lambda t: 100 * D[t]["nirnay"]["escalated"], lambda t: 100 * N[t]["nirnay"]["escalated"],
                     lambda t: f"{100 * D[t]['nirnay']['escalated']:.0f}%", lambda t: D[t]["nirnay"]["escalated"] < 0.12)  # fmt: skip
    captions[card({
        "kicker": "nirnay · escalation", "title": "Most decisions never need thinking mode",
        "subtitle": "Share of decisions nirnay escalated to Sarvam's thinking mode (fewer than 8 of 10 votes agreed).",
        "legend": [("with descriptions", PINK), ("names only", GREY)], "right_header": "escalated",
        "rows": rows, "xmax": 40, "fmt": pct,
        "footnote": "Clear categories mean fewer escalations, which is why descriptions also cut cost.",
    }, "12_escalation_rate")] = (
        "How often nirnay escalated to Sarvam's thinking mode.",
        "With descriptions, nirnay escalated 2–20% of decisions (names only: 2–32%); the rest were answered by the "
        "fast voting path.")  # fmt: skip

    lines = ["# nirnay benchmark charts: titles and captions", "",
             "All charts: sarvam-105b via Sarvam's hosted API, test sets, Oct 2026. "
             "1200×800 PNG plus a @2x version for retina screens.", ""]  # fmt: skip
    for name, (alt, caption) in captions.items():
        lines += [f"## {name}.png", f"**Alt text:** {alt}", "", f"**Caption:** {caption}", ""]
    (OUT / "LABELS.md").write_text("\n".join(lines), encoding="utf-8")
    zpath = OUT.parent / "nirnay_charts.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            z.write(f, f"nirnay_charts/{f.name}")
    print(f"wrote {len(captions)} charts + LABELS.md → {zpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
