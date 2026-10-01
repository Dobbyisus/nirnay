"""Baseline: Sarvam's default settings on the private support set.

Same prompt, labels and descriptions as nirnay (built with the same prompt builder), but the
request carries only `model` and `messages`, so Sarvam's defaults apply: thinking on, default
max_tokens and temperature, one answer, no confidence. The reply is scored with the same
parser nirnay uses. Conditions (names / described) are interleaved item by item, and the run
resumes from where it stopped.

Run:  python experiments/run_support_baseline.py [--limit N]
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

from tasks import SUPPORT_LABELS, SUPPORT_LABELS_DESCRIBED, SUPPORT_QUESTION  # noqa: E402

from nirnay import SarvamClient, SarvamError  # noqa: E402
from nirnay.pricing import cost_inr  # noqa: E402
from nirnay.prompt import build_messages, label_codes, label_descriptions, parse_reply  # noqa: E402

DATASET = ROOT / "data" / "private" / "support_v0.jsonl"
OUT = ROOT / "data" / "private" / "results" / "support_v0_default.jsonl"
CONDITIONS = {"names": SUPPORT_LABELS, "described": SUPPORT_LABELS_DESCRIBED}
MODEL = "sarvam-105b"


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

    client = SarvamClient(min_interval_s=1.6, timeout=120)
    with client, OUT.open("a", encoding="utf-8") as out:
        for i, (it, cond) in enumerate(todo, 1):
            choices = CONDITIONS[cond]
            codes = label_codes(choices)
            messages = build_messages(
                SUPPORT_QUESTION, codes, it["text"], descriptions=label_descriptions(choices)
            )
            rec = {
                "condition": cond,
                "id": it["id"],
                "language": it["language"],
                "split": it["split"],
                "label": it["label"],
                "alt_label": it["alt_label"],
                "difficulty": it["difficulty"],
                "model": MODEL,
                "settings": "sarvam defaults (only model + messages sent)",
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            start = time.perf_counter()
            try:
                data = client.chat({"model": MODEL, "messages": messages})
                latency_ms = (time.perf_counter() - start - client.last_paced_s) * 1000
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                code = parse_reply(msg.get("content"), codes)
                u = data.get("usage") or {}
                inp, out_tok = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                rec.update(
                    error=None,
                    best_guess=codes.get(code) if code else None,
                    valid=code is not None,
                    content=msg.get("content"),
                    reasoning_chars=len(msg.get("reasoning_content") or ""),
                    finish_reason=choice.get("finish_reason"),
                    input_tokens=inp,
                    cached_input_tokens=cached,
                    output_tokens=out_tok,
                    cost_inr=cost_inr(MODEL, inp, cached, out_tok),
                    latency_ms=round(latency_ms, 1),
                )
            except SarvamError as e:
                rec.update(error=str(e), status=e.status)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
