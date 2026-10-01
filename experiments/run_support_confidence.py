"""Run vote-based nirnay on the private support set, with and without label descriptions.

Each item is decided once per condition; conditions are interleaved item by item so both
see similar server load. Results are appended to a JSONL file as they arrive, so an
interrupted run resumes where it stopped.

Run:  python experiments/run_support_confidence.py [--limit N]
Analyse with experiments/analyze_support_confidence.py.
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

from nirnay import Decider, SarvamClient, SarvamError  # noqa: E402

DATASET = ROOT / "data" / "private" / "support_v0.jsonl"
OUT = ROOT / "data" / "private" / "results" / "support_v0_votes.jsonl"
CONDITIONS = {"names": SUPPORT_LABELS, "described": SUPPORT_LABELS_DESCRIBED}
MODEL = "sarvam-105b"
SAMPLES = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N items (for a smoke test)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")

    items = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    items = items[: args.limit] if args.limit else items
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("error") is None:
                done.add((r["condition"], r["id"]))
    todo = [(it, c) for it in items for c in CONDITIONS if (c, it["id"]) not in done]
    print(f"{len(done)} already done, {len(todo)} to run → {OUT.relative_to(ROOT)}")

    client = SarvamClient(min_interval_s=1.6)  # sarvam-105b: 40 requests/minute on Starter
    with (
        Decider(
            MODEL, samples=SAMPLES, client=client, escalate_below=None
        ) as decider,  # votes only
        OUT.open("a", encoding="utf-8") as out,
    ):
        for i, (it, cond) in enumerate(todo, 1):
            rec = {
                "condition": cond,
                "id": it["id"],
                "language": it["language"],
                "split": it["split"],
                "label": it["label"],
                "alt_label": it["alt_label"],
                "difficulty": it["difficulty"],
                "model": MODEL,
                "samples": SAMPLES,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            try:
                r = decider.decide(SUPPORT_QUESTION, CONDITIONS[cond], it["text"])
                rec.update(
                    error=None,
                    best_guess=r.best_guess,
                    raw_confidence=r.raw_confidence,
                    votes=r.votes,
                    valid=r.valid,
                    invalid_samples=r.invalid_samples,
                    raw_replies=r.raw_replies,
                    retried=r.retried,
                    input_tokens=r.usage.input_tokens,
                    cached_input_tokens=r.usage.cached_input_tokens,
                    output_tokens=r.usage.output_tokens,
                    cost_inr=r.usage.cost_inr,
                    latency_ms=round(r.latency_ms, 1),
                )
            except SarvamError as e:
                rec.update(error=str(e), status=e.status)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
