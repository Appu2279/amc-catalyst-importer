"""
Subject/topic tagging for a page cache that has none (the OCR route), as two
small steps around a text-only reader (a cheap Claude Code agent).

    # 1. One compact record per question: id, stem, options, answer. No images.
    venv/bin/python scripts/qbank-tags.py export <cache.json> <tags-in.json> [--parts=2]

    # 2. (A reader writes <tags-in>.part-N.out.json: {"<id>": {"subject": .., "topic": ..}})

    # 3. Copy the tags onto every page of each question, rejecting anything off-list.
    venv/bin/python scripts/qbank-tags.py apply <cache.json> <tags-in.json>

A question's id is its first PDF page, so it survives re-merging the cache.
The subject and topic lists are the importer's own (QBANK_SUBJECTS, QBANK_TOPICS).
"""
import glob
import importlib.util
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("qb", os.path.join(ROOT, "scripts", "qbank-pdf-to-batch.py"))
qb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qb)
from app import main as importer  # noqa: E402

RULES = (
    "Pick exactly one subject from SUBJECTS and one topic from that subject's TOPICS list "
    "(or null if none fits). Eye, ENT and orthopaedic questions are Surgery, whatever the setting. "
    "Judge by the clinical content the question tests, not only the setting it is told in."
)


def groups_with_answers(cache):
    return [g for g in qb.merge_pages(cache) if g["answer"] is not None]


def export(cache_path, out_path, parts):
    with open(cache_path) as f:
        cache = json.load(f)
    records = []
    for g in groups_with_answers(cache):
        answer = g["answer"]
        records.append({
            "id": str(g["pages"][0]),
            "stem": g["stem"][:900],
            "options": [o.get("text") for o in answer.get("options") or []],
            "answer": next((o.get("text") for o in answer.get("options") or [] if o.get("is_correct")), None),
        })
    size = -(-len(records) // parts)
    header = {"rules": RULES, "SUBJECTS": importer.QBANK_SUBJECTS, "TOPICS": importer.QBANK_TOPICS}
    with open(out_path, "w") as f:
        json.dump({**header, "count": len(records)}, f, indent=1)
    for k in range(parts):
        chunk = records[k * size:(k + 1) * size]
        with open(out_path.replace(".json", f".part-{k + 1}.json"), "w") as f:
            json.dump({**header, "questions": chunk}, f, indent=1)
    print(f"{len(records)} questions in {parts} parts next to {out_path}")


def apply(cache_path, tags_in_path):
    with open(cache_path) as f:
        cache = json.load(f)
    tags = {}
    for path in sorted(glob.glob(tags_in_path.replace(".json", ".part-*.out.json"))):
        with open(path) as f:
            tags.update(json.load(f))

    problems, tagged = [], 0
    for g in groups_with_answers(cache):
        qid = str(g["pages"][0])
        tag = tags.get(qid)
        if not tag:
            problems.append(f"{qid}: no tag")
            continue
        subject, topic = tag.get("subject"), tag.get("topic")
        if subject not in importer.QBANK_SUBJECTS:
            problems.append(f"{qid}: subject {subject!r} not in the list")
            continue
        if topic is not None and topic not in importer.QBANK_TOPICS.get(subject, []):
            problems.append(f"{qid}: topic {topic!r} not under {subject} — dropped")
            topic = None
        for page in g["pages"]:
            cache[str(page)]["subject"], cache[str(page)]["topic"] = subject, topic
        tagged += 1

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=1)
    print(f"tagged {tagged} questions; {len(problems)} problems")
    for p in problems:
        print("  -", p)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    parts = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--parts=")), "2"))
    if len(args) == 3 and args[0] == "export":
        export(args[1], args[2], parts)
    elif len(args) == 3 and args[0] == "apply":
        apply(args[1], args[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
