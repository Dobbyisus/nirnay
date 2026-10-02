"""Sample the public benchmark tasks into one format: data/bench/<task>.jsonl.

Expects the raw downloads in data/raw/ (see the README in experiments/). Sampling is
stratified by label with a fixed seed, and parallel datasets keep the SAME source items in
every language (linked by `pair`), so languages can be compared on identical content.

Each line: {id, pair, task, language, text, label, alt_label, difficulty, split, source}

Run:  python experiments/prepare_bench.py
"""

import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "bench"
SEED = 20261001


def stratified(rows, key, per_label, rng):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    picked = []
    for label in sorted(groups):
        members = groups[label]
        if len(members) < per_label:
            raise ValueError(f"only {len(members)} items for {label!r}, need {per_label}")
        picked += rng.sample(members, per_label)
    return picked


def item(task, pair, language, text, label, split, source):
    return {
        "id": f"{task}-{language.lower()}-{pair}",
        "pair": pair,
        "task": task,
        "language": language,
        "text": " ".join(text.split()),
        "label": label,
        "alt_label": None,
        "difficulty": "clear",
        "split": split,
        "source": source,
    }


def top(rng):
    """Hinglish-TOP (Apache-2.0): English query + human code-switched query, by domain."""
    out = []
    for split, fname, per_label in [("test", "test.tsv", 15), ("dev", "validation.tsv", 6)]:
        with open(RAW / "top" / fname, encoding="utf-8") as f:
            rows = [dict(r, row=i) for i, r in enumerate(csv.DictReader(f, delimiter="\t"))]
        rows = [r for r in rows if r["en_query"].strip() and r["cs_query"].strip()]
        for r in stratified(rows, lambda r: r["domain"], per_label, rng):
            pair = f"{split}{r['row']}"
            src = f"Hinglish-TOP human-annotated {fname} row {r['row']}"
            out.append(item("top", pair, "English", r["en_query"], r["domain"], split, src))
            out.append(item("top", pair, "Hinglish", r["cs_query"], r["domain"], split, src))
    return out


def massive(rng):
    """MASSIVE 1.1 (CC BY 4.0): same utterance id across locales, labelled by scenario."""
    locales = {"English": "en-US", "Hindi": "hi-IN", "Bengali": "bn-BD", "Tamil": "ta-IN"}
    data = {}
    for lang, loc in locales.items():
        with open(RAW / "massive" / "1.1" / "data" / f"{loc}.jsonl", encoding="utf-8") as f:
            data[lang] = {r["id"]: r for r in map(json.loads, f)}
    en = data["English"].values()
    out = []
    for split, part, per_label in [("test", "test", 7), ("dev", "dev", 3)]:
        rows = [r for r in en if r["partition"] == part]
        for r in stratified(rows, lambda r: r["scenario"], per_label, rng):
            for lang in locales:
                x = data[lang][r["id"]]
                assert x["scenario"] == r["scenario"]
                src = f"MASSIVE 1.1 {locales[lang]} id {r['id']}"
                out.append(item("massive", r["id"], lang, x["utt"], x["scenario"], split, src))
    return out


def reviews(rng):
    """IndicSentiment: product reviews; the same row in each language file is parallel."""
    langs = {"Hindi": "hi", "Bengali": "bn", "Tamil": "ta"}
    out = []
    for split, part, per_label in [("test", "test", 60), ("dev", "validation", 24)]:
        files = {}
        for lang, code in langs.items():
            with open(RAW / "indicsentiment" / f"{part}_{code}.json", encoding="utf-8") as f:
                files[lang] = [json.loads(line) for line in f if line.strip()]
        n = len(files["Hindi"])
        assert all(len(v) == n for v in files.values())
        rows = []
        for i in range(n):
            group = {lang: files[lang][i] for lang in langs}
            labels = {g["LABEL"] for g in group.values()}
            english = {g["ENGLISH REVIEW"] for g in group.values()}
            if len(labels) == 1 and None not in labels and len(english) == 1:
                rows.append((i, group))
        for i, group in stratified(rows, lambda r: r[1]["Hindi"]["LABEL"], per_label, rng):
            label = group["Hindi"]["LABEL"].lower()
            src = f"IndicSentiment {part} row {i}"
            pair = f"{split}{i}"
            out.append(
                item(
                    "reviews", pair, "English", group["Hindi"]["ENGLISH REVIEW"], label, split, src
                )
            )
            for lang in langs:
                out.append(
                    item("reviews", pair, lang, group[lang]["INDIC REVIEW"], label, split, src)
                )
    return out


def _fix_mojibake(token):
    """Undo UTF-8 that was decoded as Windows-1252 ("ðŸ™\\x8f" -> "🙏").

    Characters are mapped back to bytes one by one: through cp1252 where it has the character,
    else as raw latin-1 bytes (cp1252 leaves 0x81/0x8D/0x8F/0x90/0x9D undefined, and those
    survive as control characters). Text that isn't mojibake is returned unchanged.
    """
    raw = bytearray()
    for ch in token:
        try:
            raw += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) > 255:
                return token
            raw.append(ord(ch))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return token


def _read_conll(path):
    """Yield (uid, label or None, text) for each tweet in a SentiMix CoNLL file."""
    uid, label, tokens = None, None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines() + [""]:
        parts = line.split("\t")
        if parts[0] == "meta":
            uid, label, tokens = parts[1], (parts[2] if len(parts) > 2 else None), []
        elif line.strip():
            tokens.append(_fix_mojibake(parts[0]))
        elif uid is not None:
            text = re.sub(r"@ (\w+)", r"@\1", " ".join(tokens))
            text = re.sub(r"(\w) _ (?=\w)", r"\1_", text)  # "@Payal _ Rohatgi" -> "@Payal_Rohatgi"
            yield uid, label, text
            uid = None


def tweets(rng):
    """SemEval-2020 Task 9 SentiMix Hinglish (CC BY 4.0): real code-mixed tweets."""
    base = RAW / "sentimix" / "Semeval_2020_task9_data" / "Hinglish"
    with open(base / "Hinglish_test_labels.txt", encoding="utf-8") as f:
        test_labels = {r["Uid"]: r["Sentiment"] for r in csv.DictReader(f)}
    test = [
        (uid, test_labels[uid], text)
        for uid, _, text in _read_conll(base / "Hinglish_test_unalbelled_conll_updated.txt")
        if uid in test_labels
    ]
    dev = list(_read_conll(base / "Hinglish_dev_3k_split_conll.txt"))
    out = []
    for split, rows, per_label in [("test", test, 50), ("dev", dev, 20)]:
        rows = [r for r in rows if r[2].strip() and r[1] in {"positive", "negative", "neutral"}]
        for uid, label, text in stratified(rows, lambda r: r[1], per_label, rng):
            src = f"SentiMix Hinglish {split} uid {uid}"
            out.append(item("tweets", f"{split}{uid}", "Hinglish", text, label, split, src))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in [
        ("top", top),
        ("massive", massive),
        ("reviews", reviews),
        ("tweets", tweets),
    ]:
        rows = build(random.Random(SEED))
        assert len({r["id"] for r in rows}) == len(rows), f"{name}: duplicate ids"
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = defaultdict(int)
        for r in rows:
            counts[(r["split"], r["language"])] += 1
        print(
            f"{name:8} {len(rows):5} items  "
            + "  ".join(f"{s}/{lang}={n}" for (s, lang), n in sorted(counts.items()))
        )


if __name__ == "__main__":
    main()
