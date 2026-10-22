"""
Turn an AMC Catalyst recall text PDF (questions, answers and explanations in
one file) into a recall import batch — no Claude credits.

    # 1. Parse the PDF; writes <out.json> and figure PNGs next to it.
    venv/bin/python scripts/recall-qbank-pdf-to-batch.py parse <pdf> <out.json>

    # 2. Push to a backend as a new recall batch.
    BACKEND_URL=http://localhost:3000 ADMIN_TOKEN=... \
      venv/bin/python scripts/recall-qbank-pdf-to-batch.py push <out.json> "<title>" [--dry-run]

Why not POST /import: that endpoint expects a questions PDF and a separate
answers PDF. These files put the answer and the per-option explanations inside
each question block, so its answers parser finds nothing.

Three block layouts are read (all start with a "Question N" line):

  QBank (Feb/April/June):          Recall (March/July/August/September):
    Question N                       <Subject> [· Difficulty]  (same line height)
    <Subject> · <Difficulty> …       Question N
    <stem> [Figure N. caption]       [IMAGE-BASED QUESTION · …]
    A. … E.                          <stem>, A. … E.
    Answer: X. <text>                Correct answer: X
    EXPLANATION                      A. <text> … X. Correct. <text> …
    A. Correct./Incorrect. <text>    Reference
    Reference: <text>                • <reference> …

  October:
    Question N [Subject] [Difficulty] [IMAGE-BASED: …]
    <stem>, A. … E.
    Answer N: X. <text>
    Why X is correct: <text>
    A. <option> – Incorrect. <text>   (the four wrong options)
    Reference: <text>

Figures are placed by position on the page: an image inside a question's block
goes with that question — in the stem if it sits before the answer line, as an
answer image (shown only after answering) if it sits after it. Where the PDF
has an answer key ("Q1: A" or "1 B"), every answer is checked against it.

The subject label is free text ("Cardiology, Genetics"); each question's
backend subject comes from `subject` in <out.json>, which parse leaves as the
label's first part — edit it before pushing if the names must match existing
subjects.
"""
import asyncio
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fitz  # noqa: E402
import httpx  # noqa: E402

from app import main as importer  # noqa: E402

QUESTION_START = re.compile(r"^Question (\d+)\b(.*)$")
OPTION_LINE = re.compile(r"^([A-E])\. (.*)$")
ANSWER_LINE = re.compile(r"^(?:Answer(?: \d+)?:|Correct answer:) ([A-E])\b\.?\s*(.*)$")
WHY_LINE = re.compile(r"^Why ([A-E]) is correct:\s*(.*)$")
LIST_ITEM = re.compile(r"^(•|\d+\.\s)")
SECTION_BREAK = re.compile(r"^(Questions \d+ to \d+|Answer key\b|Image-based questions\b)")
SKIPPED_LINE = re.compile(r"^(IMAGE-BASED QUESTION\b|Image source:|Figure for Question \d+)")
NUMBERED_CAPTION = re.compile(r"^Figure \d+[a-z]?\.")
PAGE_NUMBER = re.compile(r"^Page \d+( of \d+)?$")
SUPERSCRIPT = str.maketrans("0123456789+-−()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁽⁾ⁿ")
MIN_FIGURE_AREA = 3000  # pt²; smaller images are page logos
PAGE_MARGIN = 50  # pt from the top/bottom edge where running headers and footers sit
FIGURE_RENDER_ZOOM = 3


def line_text(line: dict) -> str:
    """A text line with superscript spans kept as superscripts — plain
    extraction turns '10⁹' into '109', which changes the number."""
    parts = []
    for span in line["spans"]:
        text = span["text"]
        if span["flags"] & 1 and text.strip():
            converted = text.strip().translate(SUPERSCRIPT)
            text = converted if all(c not in "0123456789+-()n" for c in converted) else "^" + text.strip()
        parts.append(text)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def read_items(doc) -> list:
    """Every text line and figure, in reading order, as
    (page, y, kind, value). Lines repeated on many pages are headers/footers."""
    raw, margin_lines = [], set()
    for page_number, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 1:
                box = fitz.Rect(block["bbox"])
                if box.width * box.height >= MIN_FIGURE_AREA:
                    raw.append((page_number, box.y0, box.x0, "figure", box))
                continue
            for line in block["lines"]:
                text = line_text(line)
                if text:
                    raw.append((page_number, line["bbox"][1], line["bbox"][0], "text", text))
                    if line["bbox"][1] < PAGE_MARGIN or line["bbox"][3] > page.rect.height - PAGE_MARGIN:
                        margin_lines.add((page_number, text))

    # Running headers/footers: the same line in the page margin on many pages.
    # ("Reference" also repeats on most pages, but in the body, so it stays.)
    per_page = Counter(text for _, text in margin_lines)
    repeated = {text for text, n in per_page.items() if n > doc.page_count * 0.3}

    items = []
    for page_number, y, x, kind, value in sorted(raw, key=lambda r: (r[0], round(r[1]), r[2])):
        is_margin = (page_number, value) in margin_lines
        if kind == "text" and is_margin and (value in repeated or PAGE_NUMBER.match(value)):
            continue
        items.append((page_number, y, kind, value))
    return items


def join_wrapped(lines: list) -> str:
    """Rejoin lines the PDF wrapped mid-paragraph. A line ending in '-' was
    split at a hyphenated word ('first-' + 'line'); list items keep their own
    line."""
    text = ""
    for line in lines:
        if not text:
            text = line
        elif LIST_ITEM.match(line):
            text += "\n" + line
        elif text.endswith("-"):
            text += line
        else:
            text += " " + line
    return text.strip()


def split_lettered(lines: list, letters: str) -> dict:
    """Group 'A. …' lines (only the expected letters, in order) plus their
    wrapped continuations."""
    groups, order = {}, []
    remaining = list(letters)
    for line in lines:
        match = OPTION_LINE.match(line)
        if match and remaining and match.group(1) == remaining[0]:
            remaining.pop(0)
            order.append(match.group(1))
            groups[match.group(1)] = [match.group(2)]
        elif order:
            groups[order[-1]].append(line)
        else:
            raise ValueError(f"text before option {letters[0]}: {line!r}")
    if remaining:
        raise ValueError(f"missing options {remaining}")
    return {key: join_wrapped(value) for key, value in groups.items()}


def drop_captions(texts: list) -> list:
    """Remove 'Figure 54b. …' captions, including a wrapped second line —
    a caption runs until a line ending in '.' (QBank layout only; October's
    unpunctuated 'Figure for Question N' captions are one line)."""
    kept, in_caption = [], False
    for t in texts:
        if NUMBERED_CAPTION.match(t):
            in_caption = not t.endswith(".")
            continue
        if in_caption and not OPTION_LINE.match(t):
            in_caption = not t.endswith(".")
            continue
        in_caption = False
        kept.append(t)
    return kept


def parse_header(first_line_rest: str, before: list, after: list):
    """Returns (label, difficulty, flags, lines consumed from `after`)."""
    tags = re.findall(r"\[([^\]]+)\]?", first_line_rest)
    if tags:  # October: Question N [Subject] [Difficulty] [IMAGE-BASED: …]
        consumed = 0
        if first_line_rest.count("[") > first_line_rest.count("]"):
            consumed = 1  # the IMAGE-BASED tag wrapped onto the next line
        label = tags[0]
        difficulty = next((t.lower() for t in tags if t in ("Easy", "Medium", "Hard")), None)
        return label, difficulty, tags[1:], consumed
    header = before if before else after[0]
    consumed = 0 if before else 1
    label, *flags = [part.strip() for part in header.split("·")]
    difficulty = next((f.lower() for f in flags if f in ("Easy", "Medium", "Hard")), None)
    return label, difficulty, flags, consumed


def parse_block(number: int, rest: str, header_before: str, block: list) -> dict:
    texts = [value for _, _, kind, value in block if kind == "text"]
    label, difficulty, flags, consumed = parse_header(rest, header_before, texts)
    texts = drop_captions([t for t in texts[consumed:] if not SKIPPED_LINE.match(t)])

    answer_at = next(i for i, t in enumerate(texts) if ANSWER_LINE.match(t))
    answer_key = ANSWER_LINE.match(texts[answer_at]).group(1)
    options_at = next(i for i, t in enumerate(texts) if OPTION_LINE.match(t) and t.startswith("A. "))
    stem = join_wrapped(texts[:options_at])
    option_texts = split_lettered(texts[options_at:answer_at], "ABCDE")
    letters = "".join(option_texts)

    tail = texts[answer_at + 1:]
    # A long "Answer: X. …" line wraps; its tail runs until the explanations start.
    body_start = next(
        (i for i, t in enumerate(tail) if t == "EXPLANATION" or t.startswith("Key point.") or WHY_LINE.match(t) or OPTION_LINE.match(t)),
        0,
    )
    tail = tail[body_start:]
    section_at = next((i for i, t in enumerate(tail) if SECTION_BREAK.match(t)), len(tail))
    reference_at = next((i for i, t in enumerate(tail[:section_at]) if t == "Reference" or t.startswith("Reference:")), section_at)
    reference_lines = tail[reference_at:section_at]
    body = tail[:reference_at]
    if body and body[0] == "EXPLANATION":
        body = body[1:]
    key_point = ""
    if body and body[0].startswith("Key point."):
        explanations_at = next(i for i, t in enumerate(body) if OPTION_LINE.match(t))
        key_point = join_wrapped(body[:explanations_at])
        body = body[explanations_at:]

    why = WHY_LINE.match(body[0]) if body else None
    if why:  # October: the correct option's reasoning comes first, unlettered
        if why.group(1) != answer_key:
            raise ValueError(f"Q{number}: 'Why {why.group(1)}' disagrees with answer {answer_key}")
        first_wrong = next(i for i, t in enumerate(body) if OPTION_LINE.match(t))
        correct_explanation = join_wrapped([why.group(2)] + body[1:first_wrong])
        explanations = split_lettered(body[first_wrong:], letters.replace(answer_key, ""))
        explanations[answer_key] = correct_explanation
    else:
        explanations = split_lettered(body, letters)

    options = []
    for key, option_text in option_texts.items():
        explanation = explanations[key]
        verdict = re.match(r"^(?:.*?\s[–—-]\s)?(Correct|Incorrect)\.\s*", explanation)
        if verdict and verdict.start(1) == 0:
            explanation = explanation[verdict.end():]
            if (verdict.group(1) == "Correct") != (key == answer_key):
                raise ValueError(f"Q{number}: answer {answer_key} disagrees with option {key}'s verdict")
        elif verdict:  # "Option – Incorrect. reason" (October)
            explanation = explanation[:verdict.start(1)].rstrip(" –—-") + " — " + explanation[verdict.end():]
            if verdict.group(1) == "Correct" and key != answer_key:
                raise ValueError(f"Q{number}: option {key} marked Correct but answer is {answer_key}")
        options.append({
            "option_key": key,
            "option_text": option_text,
            "is_correct": key == answer_key,
            "explanation": explanation.strip() or None,
        })

    reference = ""
    if reference_lines:
        first = reference_lines[0]
        reference_text = join_wrapped(([first[len("Reference:"):].strip()] if first != "Reference" else []) + reference_lines[1:])
        separator = "\n" if reference_text.startswith("•") else " "
        reference = f"Reference:{separator}{reference_text}" if reference_text else ""
    correct_explanation = next(o["explanation"] for o in options if o["is_correct"]) or ""

    answer_y = next(y for p, y, kind, value in block if kind == "text" and ANSWER_LINE.match(value))
    answer_page = next(p for p, y, kind, value in block if kind == "text" and ANSWER_LINE.match(value))
    figures = [(p, y, box) for p, y, kind, box in block if kind == "figure"]
    return {
        "question_number": number,
        "question_text": stem,
        "explanation": "\n\n".join(t for t in (correct_explanation, key_point, reference) if t),
        "difficulty": difficulty,
        "source_type": "recall",
        "question_type": "single_choice",
        "marks": 1,
        "negative_marks": 0,
        "subject": re.split(r"[,–—·]| - ", label)[0].strip(),
        "options": options,
        "_label": label,
        "_is_image_based": any("image" in f.lower() for f in flags) or "IMAGE-BASED QUESTION" in " ".join(v for _, _, k, v in block if k == "text"),
        "_stem_figures": [(p, box) for p, y, box in figures if (p, y) < (answer_page, answer_y)],
        "_answer_figures": [(p, box) for p, y, box in figures if (p, y) >= (answer_page, answer_y)],
    }


def is_blank(doc, page_number: int, box) -> bool:
    """Some PDFs place filler images that render as plain white (sometimes
    with a thin grey frame, so 'every pixel white' is too strict)."""
    pix = doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(0.5, 0.5), clip=box, colorspace=fitz.csGRAY)
    if not pix.samples:
        return True
    return sum(1 for value in pix.samples if value > 245) / len(pix.samples) > 0.97


def merge_figures(doc, figures: list) -> list:
    """One figure per page: pieces printed side by side (a two-view X-ray,
    a composite photo) are rendered as the area covering all of them."""
    merged = {}
    for page_number, box in figures:
        if is_blank(doc, page_number, box):
            continue
        merged[page_number] = merged[page_number] | box if page_number in merged else fitz.Rect(box)
    return sorted(merged.items())


def read_answer_key(items: list) -> dict:
    texts = [value for _, _, kind, value in items if kind == "text"]
    key = {}
    for t in texts:
        match = re.fullmatch(r"Q?(\d+)[:\s]\s*([A-E])", t)
        if match:
            key.setdefault(int(match.group(1)), match.group(2))
    return key


def parse(pdf_path: str, out_path: str):
    doc = fitz.open(pdf_path)
    items = read_items(doc)
    starts = [i for i, (_, _, kind, value) in enumerate(items) if kind == "text" and QUESTION_START.match(value)]

    questions, problems, seen_blocks = [], [], {}
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(items)
        page, y, _, value = items[start]
        number, rest = int(QUESTION_START.match(value).group(1)), QUESTION_START.match(value).group(2)
        # "Recall" layout prints the subject on the Question line's height,
        # sorted just before it; it belongs to this question, not the last one.
        header_before = ""
        if start > 0 and items[start - 1][0] == page and abs(items[start - 1][1] - y) < 2 and items[start - 1][2] == "text":
            header_before = items[start - 1][3]
        block = list(items[start + 1:end])
        if n + 1 < len(starts):
            next_page, next_y, _, _ = items[end]
            if end > 0 and items[end - 1][0] == next_page and abs(items[end - 1][1] - next_y) < 2:
                block = block[:-1]  # the next question's header line
        block_text = [value for _, _, kind, value in block if kind == "text"]
        if number in seen_blocks:
            if seen_blocks[number] == block_text:
                continue  # the same page printed twice in the PDF
            problems.append(f"Q{number} (page {page}): number repeats with different content")
            continue
        seen_blocks[number] = block_text
        try:
            questions.append(parse_block(number, rest, header_before, block))
        except Exception as e:  # noqa: BLE001 — collect every problem in one pass
            problems.append(f"Q{number} (page {page}): {e}")
    if problems:
        sys.exit("could not parse:\n  " + "\n  ".join(problems))

    numbers = [q["question_number"] for q in questions]
    if numbers != list(range(1, len(numbers) + 1)):
        sys.exit(f"question numbers are not 1..N in order: {numbers}")
    answer_key = read_answer_key(items)
    if answer_key:
        mismatched = [
            q["question_number"] for q in questions
            if answer_key.get(q["question_number"]) != next(o["option_key"] for o in q["options"] if o["is_correct"])
        ]
        if mismatched:
            sys.exit(f"answer key disagrees for: {mismatched}")

    figures_dir = os.path.splitext(out_path)[0] + "-figures"
    os.makedirs(figures_dir, exist_ok=True)
    for q in questions:
        for side in ("stem", "answer"):
            files = []
            for k, (page_number, box) in enumerate(merge_figures(doc, q.pop(f"_{side}_figures"))):
                path = os.path.join(figures_dir, f"q{q['question_number']}-{side}-{k + 1}.png")
                doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(FIGURE_RENDER_ZOOM, FIGURE_RENDER_ZOOM), clip=box).save(path)
                files.append(path)
            q[f"_{side}_figure_files"] = files
        if q["_is_image_based"] and not (q["_stem_figure_files"] or q["_answer_figure_files"]):
            print(f"  Q{q['question_number']}: marked image-based but no figure found")

    with open(out_path, "w") as f:
        json.dump(questions, f, ensure_ascii=False, indent=1)
    stem_figures = sum(len(q["_stem_figure_files"]) for q in questions)
    answer_figures = sum(len(q["_answer_figure_files"]) for q in questions)
    print(f"{len(questions)} questions, {stem_figures} stem figures, {answer_figures} answer figures, "
          f"answer key {'matches all' if answer_key else 'not present'} → {out_path}")


async def upload(http, headers, path: str) -> str:
    with open(path, "rb") as image:
        resp = await http.post(
            f"{importer.BACKEND_URL}/api/admin/import-batches/images",
            files={"image": (os.path.basename(path), image.read(), "image/png")},
            headers=headers,
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()["path"]


async def push(questions_path: str, title: str, is_dry_run: bool):
    with open(questions_path) as f:
        questions = json.load(f)
    subjects = Counter(q["subject"] for q in questions)
    print(f"{title}: {len(questions)} questions, "
          f"{sum(bool(q['_stem_figure_files']) for q in questions)} with stem figures, "
          f"{sum(bool(q['_answer_figure_files']) for q in questions)} with answer figures")
    print("subjects:", dict(subjects.most_common()))
    if is_dry_run:
        print(f"dry run — nothing sent to {importer.BACKEND_URL}")
        return

    headers = {"Authorization": f"Bearer {importer.ADMIN_TOKEN}"}
    print(f"pushing to {importer.BACKEND_URL}")
    async with httpx.AsyncClient() as http:
        batch_id = await importer.create_batch(http, title)
        print(f"created batch {batch_id}")
        for q in questions:
            if q["_stem_figure_files"]:
                q["images"] = [await upload(http, headers, p) for p in q["_stem_figure_files"]]
                q["question_type"] = "image_based"
            if q["_answer_figure_files"]:
                q["answer_images"] = [await upload(http, headers, p) for p in q["_answer_figure_files"]]
            for key in [k for k in q if k.startswith("_")]:
                del q[key]

        resp = await http.post(
            f"{importer.BACKEND_URL}/api/admin/import-batches/{batch_id}/receive",
            json={"status": "success", "total_questions": len(questions), "questions": questions},
            headers=headers,
            timeout=300,
        )
        resp.raise_for_status()
        print(f"batch {batch_id}: {len(questions)} questions received")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) == 3 and args[0] == "parse":
        parse(args[1], args[2])
    elif len(args) == 3 and args[0] == "push":
        asyncio.run(push(args[1], args[2], "--dry-run" in sys.argv))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
