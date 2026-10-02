"""Charts for the write-up and README: "nirnay" vs "Sarvam default settings".

Reads experiments/results/bench_summary.json (run bench_analyze.py first) plus the raw
result files for the reliability and escalation charts. Writes PNGs to docs/charts/.
Tasks without results yet are skipped, so this can be run mid-benchmark.

Run:  python experiments/make_charts.py
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from bench_analyze import load  # noqa: E402

from nirnay.calibration import IsotonicCalibrator  # noqa: E402

OUT = ROOT / "docs" / "charts"
SUMMARY = ROOT / "experiments" / "results" / "bench_summary.json"

# Validated with the dataviz palette checker (light surface): CVD ΔE 24.7, all checks pass.
NIRNAY, DEFAULT = "#2a78d6", "#eb6834"
NAMES = {"nirnay": "nirnay", "default": "Sarvam default settings"}
COLORS = {"nirnay": NIRNAY, "default": DEFAULT}
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
SHORT = {
    "support": "Support routing\n(Hindi, Hinglish)",
    "top": "Voice domain\n(English, Hinglish)",
    "massive": "Voice scenario\n(En, Hi, Bn, Ta)",
    "reviews": "Review sentiment\n(En, Hi, Bn, Ta)",
    "tweets": "Tweet sentiment\n(Hinglish)",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
})  # fmt: skip


def style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def legend(fig, systems=("nirnay", "default"), y=0.0):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in systems]
    fig.legend(handles, [NAMES[s] for s in systems], loc="lower center", ncol=len(systems),
               frameon=False, bbox_to_anchor=(0.5, y), fontsize=10)  # fmt: skip


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


MIN_TEST_ANSWERS = 30


def enough(s):
    """Both systems have at least MIN_TEST_ANSWERS test answers for this task/condition."""
    return all((s.get(k) or {}).get("n", 0) >= MIN_TEST_ANSWERS for k in ("nirnay", "default"))


def get(summary, task, condition):
    return next((s for s in summary if s["task"] == task and s["condition"] == condition), None)


def hero(summary):
    """Support routing with descriptions: accuracy, cost and typical response time."""
    s = get(summary, "support", "described")
    if not s or not s["default"]:
        return
    panels = [
        ("Accuracy", lambda x: 100 * x["accuracy"], "{:.1f}%", (0, 100)),
        ("₹ per 1,000 decisions", lambda x: x["cost_per_1k"], "₹{:.2f}", None),
        ("Typical response time (ms)", lambda x: x["latency_p50"], "{:,.0f} ms", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (title, f, fmt, lim) in zip(axes, panels, strict=True):
        systems = ["default", "nirnay"]
        vals = [f(s[k]) for k in systems]
        bars = ax.bar([NAMES[k] for k in systems], vals, width=0.5,
                      color=[COLORS[k] for k in systems])  # fmt: skip
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center",
                    va="bottom", fontsize=10, color=INK, fontweight="bold")  # fmt: skip
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks([])
        if lim:
            ax.set_ylim(*lim)
        else:
            ax.set_ylim(0, max(vals) * 1.18)
        style(ax, "y")
    fig.suptitle(
        "Support-message routing in Hindi and Hinglish (242 test messages, 11 categories)",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )  # fmt: skip
    legend(fig, ("default", "nirnay"), y=-0.06)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "1_support_headline.png")


def by_task(summary, metric, title, fmt, fname, condition="described", scale=1.0, xlim=None):
    rows = [s for s in summary if s["condition"] == condition and s["default"]]
    if not rows:
        return
    rows.sort(key=lambda s: list(SHORT).index(s["task"]))
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(rows) + 1.2))
    h = 0.36
    for i, s in enumerate(rows):
        for j, k in enumerate(["default", "nirnay"]):
            v = s[k][metric] * scale
            y = i + (j - 0.5) * (h + 0.04)
            ax.barh(y, v, height=h, color=COLORS[k])
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)), [SHORT[s["task"]] for s in rows], color=INK2)
    ax.invert_yaxis()
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, max(s[k][metric] for s in rows for k in COLORS) * scale * 1.2)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    style(ax, "x")
    legend(fig, ("default", "nirnay"), y=-0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, fname)


def descriptions_effect(summary):
    """Accuracy without vs with label descriptions, per task, for both systems."""
    tasks = [t for t in SHORT if get(summary, t, "names") and get(summary, t, "described")]
    tasks = [t for t in tasks if get(summary, t, "names")["default"]]
    if not tasks:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.7 * len(tasks) + 1.6), sharey=True)
    for ax, k in zip(axes, ["default", "nirnay"], strict=True):
        for i, t in enumerate(tasks):
            a = 100 * get(summary, t, "names")[k]["accuracy"]
            b = 100 * get(summary, t, "described")[k]["accuracy"]
            ax.plot([a, b], [i, i], color=COLORS[k], linewidth=2, solid_capstyle="round")
            ax.scatter(
                [a], [i], s=60, facecolor=SURFACE, edgecolor=COLORS[k], linewidth=2, zorder=3
            )
            ax.scatter([b], [i], s=60, color=COLORS[k], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(max(a, b) + 1.5, i, f"{b - a:+.1f} pts", va="center", fontsize=9, color=INK2)
        ax.set_title(NAMES[k], loc="left", fontsize=11, color=INK)
        ax.set_xlim(40, 108)
        ax.set_xlabel("Accuracy (%)")
        style(ax, "x")
    axes[0].set_yticks(range(len(tasks)), [SHORT[t] for t in tasks], color=INK2)
    axes[0].invert_yaxis()
    fig.suptitle("Label descriptions: accuracy with names only (○) and with descriptions (●)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "5_label_descriptions.png")


def reliability(data):
    """Is nirnay's confidence honest? Calibrated confidence vs actual accuracy, all tasks."""
    points = []
    for d in data.values():
        cal_rows = [r for r in d["calib"] if r["condition"] == "described"]
        test = [r for r in d["nirnay"] if r["condition"] == "described" and r["split"] == "test"
                and not r["escalated"] and r["raw_confidence"] is not None]  # fmt: skip
        if not cal_rows or not test:
            continue
        cal = IsotonicCalibrator().fit(
            [r["raw_confidence"] for r in cal_rows], [r["winner"] == r["label"] for r in cal_rows]
        )
        points += [(cal(r["raw_confidence"]), r["best_guess"] == r["label"]) for r in test]
    if not points:
        return
    edges = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges, edges[1:], strict=False):
        b = [(c, ok) for c, ok in points if lo <= c < hi]
        if len(b) >= 10:
            xs.append(100 * statistics.mean(c for c, _ in b))
            ys.append(100 * sum(ok for _, ok in b) / len(b))
            ns.append(len(b))
    lo = max(0, 10 * int(min(xs + ys) // 10) - 10)  # keep every point inside the frame
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 100], [0, 100], color=AXIS, linewidth=1)
    ax.text(lo + 3, 98, "above the line: right more\noften than it claims", color=MUTED,
            fontsize=9, va="top")  # fmt: skip
    ax.text(98, lo + 4, "below the line: overconfident", color=MUTED, fontsize=9, ha="right")
    ax.plot(xs, ys, color=NIRNAY, linewidth=2)
    ax.scatter(xs, ys, s=[18 + n / 6 for n in ns], color=NIRNAY, edgecolor=SURFACE,
               linewidth=2, zorder=3)  # fmt: skip
    for x, y, n in zip(xs, ys, ns, strict=True):
        ax.text(x + 1.5, y - 3.5, f"n={n}", fontsize=8, color=INK2)
    ax.set_xlim(lo, 101)
    ax.set_ylim(lo, 101)
    ax.set_aspect("equal")
    ax.set_xlabel("Confidence nirnay reported (%)")
    ax.set_ylabel("How often it was actually right (%)")
    ax.set_title("How honest is nirnay's confidence?\nAll five tasks, decisions it didn't escalate",
                 loc="left", fontsize=11.5, fontweight="bold")  # fmt: skip
    style(ax, "both")
    fig.tight_layout()
    save(fig, "6_confidence_honesty.png")


def by_language(summary):
    """Accuracy per language, default vs nirnay, for multilingual tasks (with descriptions)."""
    tasks = [t for t in ("support", "top", "massive", "reviews")
             if (s := get(summary, t, "described")) and s["default"]
             and len(s["by_language"]) > 1]  # fmt: skip
    if not tasks:
        return
    # A dot plot (not bars from zero) so few-point gaps are visible; the axis is labelled.
    rows = []
    for t in tasks:
        for lang, v in get(summary, t, "described")["by_language"].items():
            rows.append((SHORT[t].split("\n")[0], lang, 100 * v["default_accuracy"],
                         100 * v["nirnay_accuracy"], v["n"]))  # fmt: skip
    lo = 5 * int(min(min(r[2], r[3]) for r in rows) // 5) - 5
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.4))
    for i, (_, _, d, n, _) in enumerate(rows):
        ax.plot([d, n], [i, i], color=AXIS, linewidth=1.5, zorder=1)
        # The default dot is drawn larger, so a tie shows as an orange ring around blue.
        ax.scatter([d], [i], s=120, color=DEFAULT, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([n], [i], s=45, color=NIRNAY, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.text(101.5, i, f"{n - d:+.1f} pts", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)), [f"{task} · {lang} (n={c})" for task, lang, _, _, c in rows],
                  color=INK2, fontsize=9)  # fmt: skip
    ax.invert_yaxis()
    ax.set_xlim(lo, 100.5)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Accuracy by language (with label descriptions)", loc="left", fontsize=12,
                 fontweight="bold")  # fmt: skip
    style(ax, "x")
    legend(fig, ("default", "nirnay"), y=-0.02)
    fig.tight_layout(rect=(0, 0.04, 0.93, 1))
    save(fig, "7_accuracy_by_language.png")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    # Leave out tasks that are still running: a handful of answers would chart as 0% or 100%.
    complete = [s for s in summary if enough(s)]
    for s in summary:
        if not enough(s):
            print(f"skipped {s['task']}/{s['condition']}: too few test answers so far")
    summary = complete
    hero(summary)
    by_task(summary, "accuracy", "Accuracy by task (with label descriptions)", "{:.1f}%",
            "2_accuracy_by_task.png", scale=100, xlim=(0, 112))  # fmt: skip
    by_task(summary, "cost_per_1k", "Cost: ₹ per 1,000 decisions (with label descriptions)",
            "₹{:.2f}", "3_cost_by_task.png")  # fmt: skip
    by_task(summary, "latency_p50", "Typical response time in ms (with label descriptions)",
            "{:,.0f} ms", "4_latency_by_task.png")  # fmt: skip
    descriptions_effect(summary)
    reliability(load())
    by_language(summary)


if __name__ == "__main__":
    main()
