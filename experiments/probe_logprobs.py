"""Probe Sarvam's chat API for what nirnay's confidence feature needs.

Answers (see docs/reference.md §3):
  - Is reasoning really off with reasoning_effort=None, and what does it save?
  - Do logprobs come back (V2 only), how many alternatives, are A/B/C/D single tokens?
  - How stable are the probabilities across identical calls? Does temperature change them?
  - Where does the letter sit in strict-schema mode, and do logprobs still work there?
  - Backup plan: does n-sampling (self-consistency) work, and is the prompt billed once?

Every request and response is appended to experiments/results/probe_<timestamp>.jsonl
(the API key is never logged). About 50 calls, well under ₹1.

Run:  python experiments/probe_logprobs.py
"""

import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
BASE = "https://api.sarvam.ai"
MIN_GAP_S = 1.6  # stay under 40 requests/minute (sarvam-105b, Starter tier)

CHOICES = {"A": "billing", "B": "refund", "C": "technical", "D": "other"}
SYSTEM = (
    "You route customer support messages. Categories: A = billing, B = refund, "
    "C = technical, D = other. Reply with exactly one letter: A, B, C or D."
)

# (name, message, expected letter or None if genuinely ambiguous)
MESSAGES = [
    ("hinglish_refund", "Mera paisa do baar kat gaya, refund kab milega?", "B"),
    ("hinglish_vague", "kuch gadbad hai, dekho please", None),
    ("english_technical", "The app crashes every time I open the settings page.", "C"),
    ("english_billing", "Why was I charged 499 instead of 299 on my last invoice?", "A"),
    ("hindi_refund", "मुझे अपना पैसा वापस चाहिए, ऑर्डर कैंसल कर दिया था।", "B"),
    ("hindi_technical", "ऐप में लॉगिन नहीं हो रहा, OTP नहीं आ रहा।", "C"),
    ("hinglish_billing", "bill mein extra GST kyun laga hai?", "A"),
    ("hinglish_other", "aapka office kitne baje khulta hai?", "D"),
    ("hinglish_ambiguous", "payment fail ho gaya par paisa kat gaya", None),
    ("english_other", "Do you have a franchise opportunity in Pune?", "D"),
]

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(CHOICES)}},
            "required": ["label"],
            "additionalProperties": False,
        },
    },
}


class Probe:
    def __init__(self, key: str, log_path: Path):
        self.client = httpx.Client(base_url=BASE, timeout=90, headers={"api-subscription-key": key})
        self.log = log_path.open("a", encoding="utf-8")
        self.last = 0.0
        self.calls = 0

    def post(self, test: str, path: str, body: dict):
        for attempt in range(4):
            wait = MIN_GAP_S - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            t0 = time.perf_counter()
            try:
                r = self.client.post(path, json=body)
                status, text = r.status_code, r.text
            except httpx.HTTPError as e:
                status, text = None, f"{type(e).__name__}: {e}"
            latency_ms = round((time.perf_counter() - t0) * 1000)
            self.calls += 1
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                data = None
            self.log.write(
                json.dumps(
                    {
                        "test": test,
                        "path": path,
                        "request": body,
                        "status": status,
                        "latency_ms": latency_ms,
                        "response": data if data is not None else text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.log.flush()
            if status in (429, 503) and attempt < 3:
                time.sleep(2**attempt * 5)
                continue
            return status, data, text, latency_ms
        return status, data, text, latency_ms


def chat_body(model, message, *, reasoning_off=True, max_tokens=5, **extra):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}],
        "max_tokens": max_tokens,
    }
    if reasoning_off:
        body["reasoning_effort"] = None
    body.update(extra)
    return body


def err(text):
    return (text or "")[:300].replace("\n", " ")


def letter_distribution(choice):
    """Find the first output position that is an answer letter.

    Returns its token, the distribution over letters, the probability mass on letters and the
    number of alternatives returned."""
    lp = (choice or {}).get("logprobs")
    content = lp.get("content") if isinstance(lp, dict) else None
    if not content:
        return None
    for pos in content:
        tok = str(pos.get("token", "")).strip().strip('"')
        if tok in CHOICES:
            alts = pos.get("top_logprobs") or [{"token": pos["token"], "logprob": pos["logprob"]}]
            probs = Counter()
            for a in alts:
                t = str(a.get("token", "")).strip().strip('"')
                if t in CHOICES:
                    probs[t] += math.exp(a["logprob"])
            mass = sum(probs.values())
            dist = {k: probs[k] / mass for k in CHOICES} if mass else {}
            return {
                "token_repr": repr(pos["token"]),
                "dist": dist,
                "letter_mass": mass,
                "n_alts": len(alts),
                "alt_tokens": [a.get("token") for a in alts],
            }
    return {
        "token_repr": None,
        "dist": {},
        "letter_mass": 0,
        "n_alts": 0,
        "alt_tokens": [p.get("token") for p in content],
    }


def fmt_dist(d):
    return "  ".join(f"{k}:{d.get(k, 0):.2f}" for k in CHOICES) if d else "(no letter found)"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key or key == "your_key_here":
        sys.exit("SARVAM_API_KEY is missing or empty in .env — save the key there and rerun.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS / f"probe_{stamp}.jsonl"
    p = Probe(key, log_path)
    summary = {"run": stamp}
    msg0, msg1 = MESSAGES[0][1], MESSAGES[1][1]

    print("\n== 1. Reasoning: default vs off (V1, sarvam-105b) ==")
    for label, off, mt in [
        ("default (thinking on)", False, 1024),
        ("reasoning_effort=None", True, 5),
    ]:
        s, d, t, ms = p.post(
            f"v1_reasoning_{'off' if off else 'default'}",
            "/v1/chat/completions",
            chat_body("sarvam-105b", msg0, reasoning_off=off, max_tokens=mt),
        )
        if s == 403:
            sys.exit(f"403: API key rejected. Check the key in .env. {err(t)}")
        if s != 200:
            print(f"  {label}: HTTP {s} {err(t)}")
            summary[f"reasoning_{label}"] = {"status": s, "error": err(t)}
            continue
        m = d["choices"][0]["message"]
        rc = m.get("reasoning_content") or ""
        print(
            f"  {label}: answer={m.get('content')!r}  usage={d.get('usage')}  "
            f"reasoning_chars={len(rc)}  latency={ms} ms"
        )
        summary[f"reasoning_{label}"] = {
            "answer": m.get("content"),
            "usage": d.get("usage"),
            "reasoning_chars": len(rc),
            "latency_ms": ms,
        }

    print("\n== 2. Does V1 accept logprobs? (docs say V2 only) ==")
    for model in ["sarvam-105b", "sarvam-105b-conversations"]:
        body = chat_body(
            model, msg0, reasoning_off=(model == "sarvam-105b"), logprobs=True, top_logprobs=5
        )
        s, d, t, ms = p.post(f"v1_logprobs_{model}", "/v1/chat/completions", body)
        if s == 200:
            ld = letter_distribution(d["choices"][0])
            print(
                f"  {model}: HTTP 200, logprobs={'present' if ld else 'absent/null'}"
                + (f"  {fmt_dist(ld['dist'])}" if ld else "")
            )
            summary[f"v1_logprobs_{model}"] = {"status": 200, "logprobs": ld}
        else:
            print(f"  {model}: HTTP {s} {err(t)}")
            summary[f"v1_logprobs_{model}"] = {"status": s, "error": err(t)}

    print("\n== 3. V2 access + logprobs (sarvam-105b) ==")
    v2_reasoning_off = True
    s, d, t, ms = p.post(
        "v2_access",
        "/v2/chat/completions",
        chat_body("sarvam-105b", msg0, logprobs=True, top_logprobs=5),
    )
    if s == 400 and "reasoning" in t.lower():
        print(f"  reasoning_effort=None rejected on V2 ({err(t)}); retrying without it")
        v2_reasoning_off = False
        s, d, t, ms = p.post(
            "v2_access_no_reasoning_field",
            "/v2/chat/completions",
            chat_body(
                "sarvam-105b",
                msg0,
                reasoning_off=False,
                logprobs=True,
                top_logprobs=5,
                max_tokens=256,
            ),
        )
    v2_ok = s == 200
    summary["v2_access"] = {
        "status": s,
        "error": None if v2_ok else err(t),
        "reasoning_field_accepted": v2_reasoning_off,
    }
    if not v2_ok:
        print(f"  V2 NOT available for this key: HTTP {s} {err(t)}")
        print(
            "  → Request beta whitelisting from Sarvam (docs: api-reference/beta-apis). "
            "Skipping V2 tests."
        )
    else:
        ld = letter_distribution(d["choices"][0])
        print(
            f"  V2 OK. answer={d['choices'][0]['message'].get('content')!r} usage={d.get('usage')}"
        )
        print(f"  letter token={ld and ld['token_repr']}  alternatives={ld and ld['alt_tokens']}")
        print(
            f"  distribution: {fmt_dist(ld and ld['dist'])}  "
            f"(mass on letters={ld and ld['letter_mass']:.3f})"
        )
        summary["v2_first"] = {"usage": d.get("usage"), "logprobs": ld}
        mt = 5 if v2_reasoning_off else 256

        def v2(test, message, **extra):
            return p.post(
                test,
                "/v2/chat/completions",
                chat_body(
                    "sarvam-105b",
                    message,
                    reasoning_off=v2_reasoning_off,
                    max_tokens=extra.pop("max_tokens", mt),
                    **extra,
                ),
            )

        print("\n== 4. How many alternatives can we get? ==")
        best_k = 5
        summary["top_k"] = {}
        for k in [10, 20, 50]:
            s, d, t, _ = v2(f"v2_top_{k}", msg0, logprobs=True, top_logprobs=k)
            if s == 200:
                ld = letter_distribution(d["choices"][0])
                n = ld["n_alts"] if ld else 0
                print(f"  top_logprobs={k}: OK, returned {n} alternatives")
                summary["top_k"][k] = {"status": 200, "returned": n}
                best_k = k if n else best_k
            else:
                print(f"  top_logprobs={k}: HTTP {s} {err(t)}")
                summary["top_k"][k] = {"status": s, "error": err(t)}
        best_k = min(best_k, 20)

        print(f"\n== 5. All 10 messages, prompt-only (top_logprobs={best_k}) ==")
        summary["prompt_only"] = {}
        for name, msg, gold in MESSAGES:
            s, d, t, ms = v2(f"v2_prompt_{name}", msg, logprobs=True, top_logprobs=best_k)
            if s != 200:
                print(f"  {name:20s} HTTP {s} {err(t)}")
                continue
            ld = letter_distribution(d["choices"][0])
            ans = d["choices"][0]["message"].get("content")
            print(
                f"  {name:20s} ans={ans!r:6} gold={gold or '?'}  "
                f"{fmt_dist(ld and ld['dist'])}  {ms} ms"
            )
            summary["prompt_only"][name] = {
                "answer": ans,
                "gold": gold,
                "logprobs": ld,
                "usage": d.get("usage"),
                "latency_ms": ms,
            }

        print("\n== 6. Stability: same question x10 ==")
        summary["stability"] = {}
        for name, msg, _ in MESSAGES[:2]:
            tops = []
            for i in range(10):
                s, d, t, _ = v2(f"v2_repeat_{name}_{i}", msg, logprobs=True, top_logprobs=best_k)
                ld = letter_distribution(d["choices"][0]) if s == 200 else None
                if ld and ld["dist"]:
                    letter = max(ld["dist"], key=ld["dist"].get)
                    tops.append((letter, round(ld["dist"][letter], 4)))
            vals = [v for _, v in tops]
            if vals:
                print(
                    f"  {name}: top letters={Counter(letter for letter, _ in tops)}  "
                    f"top prob min={min(vals):.3f} max={max(vals):.3f} "
                    f"spread={max(vals) - min(vals):.3f}"
                )
            summary["stability"][name] = tops

        print("\n== 7. Does temperature change the logprobs? ==")
        summary["temperature"] = {}
        for temp in [0.0, 1.0]:
            s, d, t, _ = v2(
                f"v2_temp_{temp}", msg1, logprobs=True, top_logprobs=best_k, temperature=temp
            )
            ld = letter_distribution(d["choices"][0]) if s == 200 else None
            print(
                f"  temperature={temp}: " + (fmt_dist(ld["dist"]) if ld else f"HTTP {s} {err(t)}")
            )
            summary["temperature"][temp] = ld

        print("\n== 8. Strict schema mode + logprobs ==")
        summary["strict_schema"] = {}
        for name, msg, gold in MESSAGES:
            s, d, t, ms = v2(
                f"v2_schema_{name}",
                msg,
                logprobs=True,
                top_logprobs=best_k,
                response_format=SCHEMA,
                max_tokens=mt + 15,
            )
            if s != 200:
                print(f"  {name:20s} HTTP {s} {err(t)}")
                summary["strict_schema"][name] = {"status": s, "error": err(t)}
                continue
            ld = letter_distribution(d["choices"][0])
            ans = d["choices"][0]["message"].get("content")
            print(
                f"  {name:20s} content={ans!r:18} {fmt_dist(ld and ld['dist'])}  "
                f"out_tokens={(d.get('usage') or {}).get('completion_tokens')}"
            )
            summary["strict_schema"][name] = {
                "content": ans,
                "gold": gold,
                "logprobs": ld,
                "usage": d.get("usage"),
                "latency_ms": ms,
            }

    print("\n== 9. Backup plan: ask n=10 times in one call (V1) ==")
    summary["self_consistency"] = {}
    for model, name, msg in [
        ("sarvam-105b", *MESSAGES[0][:2]),
        ("sarvam-105b", *MESSAGES[1][:2]),
        ("sarvam-105b-conversations", *MESSAGES[1][:2]),
    ]:
        body = chat_body(
            model, msg, reasoning_off=(model == "sarvam-105b"), max_tokens=3, n=10, temperature=1.0
        )
        s, d, t, ms = p.post(f"v1_n10_{model}_{name}", "/v1/chat/completions", body)
        if s != 200:
            print(f"  {model} / {name}: HTTP {s} {err(t)}")
            summary["self_consistency"][f"{model}/{name}"] = {"status": s, "error": err(t)}
            continue
        votes = Counter((c["message"].get("content") or "").strip()[:1] for c in d["choices"])
        print(
            f"  {model} / {name}: votes={dict(votes)}  choices={len(d['choices'])}  "
            f"usage={d.get('usage')}  {ms} ms"
        )
        summary["self_consistency"][f"{model}/{name}"] = {
            "votes": dict(votes),
            "usage": d.get("usage"),
            "latency_ms": ms,
        }

    summary["calls"] = p.calls
    (RESULTS / f"probe_{stamp}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nDone: {p.calls} calls. Raw log: {log_path.relative_to(ROOT)}")
    print(f"Summary: {(RESULTS / f'probe_{stamp}_summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
