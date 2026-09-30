"""Run the Decider on a handful of real messages against the live Sarvam API.

Run:  python experiments/try_decider.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nirnay import Decider, SarvamClient  # noqa: E402

QUESTION = "Which team should handle this customer support message?"
CHOICES = ["billing", "refund", "technical", "other"]

# (message, expected label or None if genuinely ambiguous)
MESSAGES = [
    ("Mera paisa do baar kat gaya, refund kab milega?", "refund"),
    ("kuch gadbad hai, dekho please", None),
    ("The app crashes every time I open the settings page.", "technical"),
    ("Why was I charged 499 instead of 299 on my last invoice?", "billing"),
    ("मुझे अपना पैसा वापस चाहिए, ऑर्डर कैंसल कर दिया था।", "refund"),
    ("ऐप में लॉगिन नहीं हो रहा, OTP नहीं आ रहा।", "technical"),
    ("bill mein extra GST kyun laga hai?", "billing"),
    ("aapka office kitne baje khulta hai?", "other"),
    ("payment fail ho gaya par paisa kat gaya", None),
    ("Do you have a franchise opportunity in Pune?", "other"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    client = SarvamClient(min_interval_s=1.6)  # stay under 40 requests/minute
    total_cost, correct, graded = 0.0, 0, 0
    with Decider("sarvam-105b", samples=10, threshold=0.7, client=client) as decider:
        for message, expected in MESSAGES:
            r = decider.decide(QUESTION, CHOICES, message)
            total_cost += r.usage.cost_inr or 0
            if expected:
                graded += 1
                correct += r.best_guess == expected
            outcome = "ESCALATE" if r.abstained else r.choice
            mark = "" if expected is None else ("✓" if r.best_guess == expected else "✗")
            print(
                f"{message[:48]:50} → {outcome!s:9} conf={r.confidence:.1f} "
                f"votes={r.votes} {mark:1}  in={r.usage.input_tokens} "
                f"cached={r.usage.cached_input_tokens} out={r.usage.output_tokens} "
                f"{r.latency_ms:.0f} ms"
            )
    print(f"\nBest guess correct on {correct}/{graded} messages with a clear answer.")
    print(f"Total cost: ₹{total_cost:.4f}  (₹{total_cost / len(MESSAGES) * 1000:.2f} per 1,000)")


if __name__ == "__main__":
    main()
