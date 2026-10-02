"""Live run of nirnay as shipped (votes + escalation to thinking) on the support test split.

Product defaults: sarvam-105b, 10 votes, escalate when fewer than 8 of 10 agree, label
descriptions, and the calibrator fitted on the dev split (for reported confidence only; it
doesn't change answers). Results are appended as they arrive and the run resumes if stopped.

Run:  python experiments/run_support_live.py [--limit N]
Analyse with experiments/analyze_support_live.py.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tasks import SUPPORT_LABELS, SUPPORT_LABELS_DESCRIBED, SUPPORT_QUESTION  # noqa: E402

from nirnay import Decider, IsotonicCalibrator, SarvamClient, SarvamError  # noqa: E402

DATASET = ROOT / "data" / "private" / "support_v0.jsonl"
RESULTS = ROOT / "data" / "private" / "results"
CONDITIONS = {"described": SUPPORT_LABELS_DESCRIBED, "names": SUPPORT_LABELS}
# Described keeps its original filename so earlier results stay where they were.
OUT = {
    "described": RESULTS / "support_v0_nirnay_live.jsonl",
    "names": RESULTS / "support_v0_nirnay_live_names.jsonl",
}
MODEL = "sarvam-105b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N test items (for a smoke test)")
    ap.add_argument("--condition", choices=CONDITIONS, default="described")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")

    out_path = OUT[args.condition]
    items = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    items = [it for it in items if it["split"] == "test"]
    items = items[: args.limit] if args.limit else items
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("error") is None:
                done.add(r["id"])
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to run → {out_path.relative_to(ROOT)}")

    client = SarvamClient(min_interval_s=1.6, timeout=120)
    calibrator = IsotonicCalibrator.load(RESULTS / f"support_v0_calibrator_{args.condition}.json")
    with (
        Decider(MODEL, calibrator=calibrator, client=client) as decider,
        out_path.open("a", encoding="utf-8") as out,
    ):
        print(f"escalate_below={decider.escalate_below}, samples={decider.samples}")
        for i, it in enumerate(todo, 1):
            rec = {
                "id": it["id"],
                "language": it["language"],
                "split": it["split"],
                "label": it["label"],
                "alt_label": it["alt_label"],
                "difficulty": it["difficulty"],
                "model": MODEL,
                "escalate_below": decider.escalate_below,
                "condition": args.condition,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            try:
                r = decider.decide(SUPPORT_QUESTION, CONDITIONS[args.condition], it["text"])
                rec.update(
                    error=None,
                    choice=r.choice,
                    best_guess=r.best_guess,
                    valid=r.valid,
                    escalated=r.escalated,
                    escalation_answer=r.escalation_answer,
                    vote_winner=max(r.votes, key=r.votes.get) if r.votes else None,
                    votes=r.votes,
                    raw_confidence=r.raw_confidence,
                    confidence=r.confidence,
                    invalid_samples=r.invalid_samples,
                    calls=r.usage.calls,
                    input_tokens=r.usage.input_tokens,
                    cached_input_tokens=r.usage.cached_input_tokens,
                    output_tokens=r.usage.output_tokens,
                    cost_inr=r.usage.cost_inr,
                    latency_ms=round(r.latency_ms, 1),
                    raw_replies=r.raw_replies,
                )
            except SarvamError as e:
                rec.update(error=str(e), status=e.status)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
