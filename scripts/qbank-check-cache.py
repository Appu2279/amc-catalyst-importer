"""
Check a QBank page cache before pushing it, and drop what must not be pushed.

    venv/bin/python scripts/qbank-check-cache.py <pages-dir> <cache.json>

Prints problems and the pages that need reading by hand; edits the cache only
to clear answer-page-only figures (they give the answer away). Exclude a page
by hand with `"_excluded": "<why>"` plus empty options.
"""
import glob
import importlib.util
import json
import os
import re
import sys
from difflib import SequenceMatcher

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("qb", os.path.join(ROOT, "scripts", "qbank-pdf-to-batch.py"))
qb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qb)
from app import main as importer  # noqa: E402
from PIL import Image  # noqa: E402

GREEN = (217, 245, 238)


def green_share(png):
    img = Image.open(png).convert("RGB")
    img.thumbnail((300, 1000))
    px = list(img.getdata())
    return sum(1 for r, g, b in px if abs(r - GREEN[0]) < 12 and abs(g - GREEN[1]) < 10 and abs(b - GREEN[2]) < 12) / len(px)


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()[:250]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pages_dir, cache_path = sys.argv[1:]
    cache = json.load(open(cache_path))
    problems, read_by_hand = [], []

    for key in sorted(cache, key=int):
        page, n = cache[key], int(key)
        if page.get("_excluded"):
            continue
        options = page.get("options") or []
        correct = [o for o in options if o.get("is_correct")]
        if page.get("has_answer") and len(correct) != 1:
            problems.append(f"page {n}: answered but {len(correct)} correct options")
        share = green_share(os.path.join(pages_dir, f"p{n}.png"))
        if not page.get("has_answer") and share > 0.004:
            read_by_hand.append((n, "green highlight present but no answer parsed"))
        if page.get("has_answer") and len(options) < 4:
            read_by_hand.append((n, f"answer page with only {len(options)} options parsed"))

    groups = qb.merge_pages(cache)
    dropped = []
    for g in groups:
        if g["figure"] is not None and g["figure"].get("has_answer"):
            dropped.append(g["pages"])
            for f in g["figures"]:
                cache[str(f["_page"])]["has_image"] = False
            g["figure"] = None
    questions, skipped = qb.build_questions(groups)

    for s in skipped:
        read_by_hand.append((s["pages"], s["reason"]))
    for q in questions:
        if len((q["question_text"] or "").strip()) < 120:
            problems.append(f"pages {q['_source_pages']}: stem is only {q['question_text']!r} — case missing?")
        if not q.get("subject"):
            problems.append(f"pages {q['_source_pages']}: no subject")
        if len(q["options"]) < 4:
            problems.append(f"pages {q['_source_pages']}: only {len(q['options'])} options")

    own = cache_path.replace(".json", ".questions.json")
    earlier = []
    for path in sorted(glob.glob(os.path.join(os.path.dirname(cache_path), "emedici-*.cache.questions.json"))):
        if path != own:
            earlier += [(os.path.basename(path), norm(q["question_text"])) for q in json.load(open(path))]
    overlaps = []
    for q in questions:
        stem = norm(q["question_text"])
        for name, other in earlier:
            if stem[:60] == other[:60] and SequenceMatcher(None, stem, other).ratio() > 0.92:
                overlaps.append((q["_source_pages"], name))
                break

    json.dump(cache, open(cache_path, "w"), indent=1)
    subjects = {}
    for q in questions:
        subjects[q.get("subject")] = subjects.get(q.get("subject"), 0) + 1
    print(f"questions {len(questions)}, figures {sum(1 for q in questions if q['_figure'])}, skipped {len(skipped)}")
    print("subjects:", subjects)
    print(f"answer-page-only figures dropped: {dropped}")
    print(f"possible repeats of earlier batches: {overlaps}")
    print(f"{len(problems)} problems:")
    for p in problems:
        print("  -", p)
    print(f"{len(read_by_hand)} to read by hand:")
    for p in read_by_hand:
        print("  -", p)


if __name__ == "__main__":
    main()
