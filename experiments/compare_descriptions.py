"""Names-only vs labels-with-descriptions on messages where label names are misleading.

Uses Bitext's 11 support categories, where "cancel" means cancellation FEES only (cancelling
an order is "order") and "subscription" means newsletters/promos only.

Run:  python experiments/compare_descriptions.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nirnay import Decider, SarvamClient  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from tasks import SUPPORT_LABELS, SUPPORT_LABELS_DESCRIBED, SUPPORT_QUESTION  # noqa: E402

QUESTION = SUPPORT_QUESTION
DESCRIBED = SUPPORT_LABELS_DESCRIBED
NAMES = SUPPORT_LABELS

# (message, expected label under Bitext's definitions)
MESSAGES = [
    ("mujhe ye order nahi chahiye, cancel kar do please", "order"),
    ("order cancel kr do yaar", "order"),
    ("मेरा ऑर्डर रद्द कर दो", "order"),
    ("cancellation pe kuch paise katenge kya?", "cancel"),
    ("aapke offers wale emails bahut aate hai, roko", "subscription"),
    ("newsletter chahiye mujhe", "subscription"),
    ("track karna hai mera parcel", "order"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    client = SarvamClient(min_interval_s=1.6)
    hits = {"names": 0, "described": 0}
    with Decider("sarvam-105b", client=client, escalate_below=None) as d:  # votes only
        for message, expected in MESSAGES:
            row = [f"{message[:44]:46} expect={expected:12}"]
            for mode, choices in [("names", NAMES), ("described", DESCRIBED)]:
                r = d.decide(QUESTION, choices, message)
                ok = r.best_guess == expected
                hits[mode] += ok
                row.append(
                    f"{mode}: {r.best_guess!s:12} {r.confidence:.1f} {'✓' if ok else '✗'}"
                    f"  in={r.usage.input_tokens}"
                )
            print("  |  ".join(row))
    n = len(MESSAGES)
    print(f"\nnames only: {hits['names']}/{n} correct   with descriptions: {hits['described']}/{n}")


if __name__ == "__main__":
    main()
