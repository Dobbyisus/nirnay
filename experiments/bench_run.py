"""Run a benchmark system on the public tasks built by prepare_bench.py.

Systems:
  nirnay   the product as shipped: 10 votes + escalation to thinking when < 8/10 agree
  default  Sarvam default settings: only model + messages are sent (thinking on), one answer

Both use the same prompt, labels and descriptions (built by nirnay's prompt builder), and
replies are scored later with the same parser. The names-only and described conditions are
interleaved item by item. Results are appended to experiments/results/bench_<task>_<system>.jsonl
as they arrive, and a rerun resumes where it stopped.

Run e.g.:
  python experiments/bench_run.py --system nirnay --tasks top massive reviews tweets
  python experiments/bench_run.py --system default --splits test --interval 4.0
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tasks import TASKS, choices  # noqa: E402

from nirnay import Decider, SarvamClient, SarvamError  # noqa: E402
from nirnay.pricing import cost_inr  # noqa: E402
from nirnay.prompt import build_messages, label_codes, label_descriptions, parse_reply  # noqa: E402

BENCH = ROOT / "data" / "bench"
RESULTS = ROOT / "experiments" / "results"
MODEL = "sarvam-105b"
PUBLIC_TASKS = ["top", "massive", "reviews", "tweets"]


def run_nirnay(decider, task, condition, text):
    r = decider.decide(TASKS[task]["question"], choices(task, condition), text)
    return {
        "best_guess": r.best_guess,
        "valid": r.valid,
        "escalated": r.escalated,
        "escalation_answer": r.escalation_answer,
        "escalation_reply": r.escalation_reply,
        "vote_winner": max(r.votes, key=r.votes.get) if r.votes else None,
        "votes": r.votes,
        "raw_confidence": r.raw_confidence,
        "invalid_samples": r.invalid_samples,
        "calls": r.usage.calls,
        "input_tokens": r.usage.input_tokens,
        "cached_input_tokens": r.usage.cached_input_tokens,
        "output_tokens": r.usage.output_tokens,
        "cost_inr": r.usage.cost_inr,
        "latency_ms": round(r.latency_ms, 1),
        "raw_replies": r.raw_replies,
    }


def run_default(client, task, condition, text):
    ch = choices(task, condition)
    codes = label_codes(ch)
    messages = build_messages(
        TASKS[task]["question"], codes, text, descriptions=label_descriptions(ch)
    )
    start = time.perf_counter()
    data = client.chat({"model": MODEL, "messages": messages})
    latency_ms = (time.perf_counter() - start - client.last_paced_s) * 1000
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    code = parse_reply(msg.get("content"), codes)
    u = data.get("usage") or {}
    inp, out = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {
        "best_guess": codes[code] if code else None,
        "valid": code is not None,
        "content": msg.get("content"),
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
        "finish_reason": choice.get("finish_reason"),
        "calls": 1,
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "output_tokens": out,
        "cost_inr": cost_inr(MODEL, inp, cached, out),
        "latency_ms": round(latency_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["nirnay", "default"], required=True)
    ap.add_argument("--tasks", nargs="+", default=PUBLIC_TASKS, choices=PUBLIC_TASKS)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"])
    ap.add_argument("--conditions", nargs="+", default=["names", "described"])
    ap.add_argument("--interval", type=float, default=1.6, help="min seconds between API calls")
    ap.add_argument("--limit", type=int, help="first N items per task (smoke test)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    RESULTS.mkdir(parents=True, exist_ok=True)

    client = SarvamClient(min_interval_s=args.interval, timeout=180)
    decider = Decider(MODEL, client=client) if args.system == "nirnay" else None
    for task in args.tasks:
        items = [json.loads(x) for x in (BENCH / f"{task}.jsonl").read_text("utf-8").splitlines()]
        items = [it for it in items if it["split"] in args.splits]
        items = items[: args.limit] if args.limit else items
        out_path = RESULTS / f"bench_{task}_{args.system}.jsonl"
        done = set()
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("error") is None:
                    done.add((r["condition"], r["id"]))
        todo = [(it, c) for it in items for c in args.conditions if (c, it["id"]) not in done]
        print(f"[{args.system}] {task}: {len(done)} done, {len(todo)} to run", flush=True)
        with out_path.open("a", encoding="utf-8") as out:
            for i, (it, cond) in enumerate(todo, 1):
                rec = {k: it[k] for k in ("id", "pair", "task", "language", "split", "label")}
                rec.update(
                    alt_label=it["alt_label"],
                    difficulty=it["difficulty"],
                    condition=cond,
                    system=args.system,
                    model=MODEL,
                    time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                try:
                    if args.system == "nirnay":
                        rec.update(run_nirnay(decider, task, cond, it["text"]), error=None)
                        rec["escalate_below"] = decider.escalate_below
                    else:
                        rec.update(run_default(client, task, cond, it["text"]), error=None)
                except SarvamError as e:
                    rec.update(error=str(e), status=e.status)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if i % 50 == 0 or i == len(todo):
                    print(f"[{args.system}] {task}: {i}/{len(todo)}", flush=True)
    client.close()


if __name__ == "__main__":
    main()
