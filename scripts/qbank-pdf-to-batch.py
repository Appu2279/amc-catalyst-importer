"""
Turn a screenshot-export QBank PDF into an import batch, in two steps, so the
Claude reads are paid for once and the same result can be pushed to any
environment (local first, then production).

    # 1. Read every page with Claude and cache the raw results (resumable).
    venv/bin/python scripts/qbank-pdf-to-batch.py extract <pdf> <cache.json>

    # 2. Merge into questions and push to a backend as a new qbank batch.
    BACKEND_URL=http://localhost:3000 ADMIN_TOKEN=... \
      venv/bin/python scripts/qbank-pdf-to-batch.py push <pdf> <cache.json> "<title>" [--dry-run]

Why not POST /import/qbank: that endpoint merges pages by question number,
which assumes one numbering run per PDF. Exports stitched together from several
eMedici sessions restart at "Question 1" part-way through, so numbers repeat
and questions would overwrite each other. Here pages are merged in page order
instead: a new question starts whenever the number changes.

Figures are cropped from the PDF and uploaded to the target backend at push
time, so each environment gets them in its own storage.
"""
import asyncio
import json
import os
import re
from collections import Counter
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fitz  # noqa: E402
import httpx  # noqa: E402

from app import main as importer  # noqa: E402


def load_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path) as f:
        return json.load(f)


def save_cache(cache_path: str, cache: dict):
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=1)
    os.replace(tmp_path, cache_path)


async def extract(pdf_path: str, cache_path: str):
    import anthropic

    if not importer.ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY is not set in the importer .env")

    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count

    cache = load_cache(cache_path)
    todo = [p for p in range(1, total_pages + 1) if str(p) not in cache]
    print(f"{total_pages} pages, {len(cache)} cached, reading {len(todo)}")

    client = anthropic.AsyncAnthropic(api_key=importer.ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(max(1, importer.QBANK_CONCURRENCY))

    async def read_page(page_number: int):
        data = await importer._extract_page(client, sem, pdf_path, page_number)
        if data is None:
            print(f"  page {page_number}: FAILED (re-run extract to retry)")
            return
        cache[str(page_number)] = data
        if len(cache) % 25 == 0:
            save_cache(cache_path, cache)
            print(f"  {len(cache)}/{total_pages}")

    await asyncio.gather(*(read_page(p) for p in todo))
    save_cache(cache_path, cache)

    failed = [p for p in range(1, total_pages + 1) if str(p) not in cache]
    print(f"done: {len(cache)}/{total_pages} pages cached" + (f", failed {failed}" if failed else ""))


def option_set(data: dict) -> set:
    return {
        re.sub(r"[^a-z0-9]+", " ", (o.get("text") or "").lower()).strip()
        for o in data.get("options") or []
    } - {""}


def is_same_question(a: set, b: set) -> bool:
    """Two pages show the same question when most of their options match. Not
    all, because a scrolled screenshot can cut an option off."""
    return bool(a and b) and len(a & b) / min(len(a), len(b)) >= 0.6


def merge_pages(cache: dict) -> list:
    """Group pages into questions.

    Question numbers can't be trusted (they restart per session, and scrolled
    screenshots lose the header), so a question is identified by its option
    texts: consecutive pages with matching options are one question — its
    unanswered page plus one or more answer screenshots. A page with no options
    (an explanation that overflowed) belongs to the question before it.

    The same question can also appear again later in the PDF; those repeats
    are folded into the first occurrence.
    """
    groups = []
    for page_number in sorted(int(p) for p in cache):
        data = cache[str(page_number)]
        options = option_set(data)
        current = groups[-1] if groups else None

        if not options:
            if current and data.get("has_answer"):
                current["overflow"].append(data)
                current["pages"].append(page_number)
            continue

        if current is None or not is_same_question(options, current["options"]):
            repeat = next((g for g in groups if is_same_question(options, g["options"])), None)
            current = repeat or {"options": set(), "pages": [], "answers": [], "stems": [], "overflow": [], "figures": []}
            if repeat is None:
                groups.append(current)
            else:
                groups.remove(repeat)
                groups.append(repeat)

        current["options"] |= options
        current["pages"].append(page_number)
        current["stems"].append((data.get("question_text") or "").strip())
        if data.get("has_answer"):
            current["answers"].append(data)
        if data.get("has_image"):
            current["figures"].append(data)

    for group in groups:
        group["pages"].sort()
        # The richest answer screenshot wins; the longest stem wins because a
        # scrolled answer page often shows only the lead-in sentence.
        group["answer"] = max(group["answers"], key=importer._score) if group["answers"] else None
        group["stem"] = max(group["stems"], key=len)
        # Crop the figure from the unanswered page when there is one: answer
        # screenshots are often scrolled down, cutting the figure off, and can
        # show an annotated explanation figure instead of the question's own.
        unanswered = [f for f in group["figures"] if not f.get("has_answer")]
        group["figure"] = (unanswered or group["figures"] or [None])[0]
        if group["answer"]:
            for extra in group["overflow"]:
                group["answer"].setdefault("key_points", []).extend(extra.get("key_points") or [])
    groups.sort(key=lambda g: g["pages"][0])
    return groups


def build_questions(groups: list) -> tuple:
    questions, skipped = [], []
    for index, group in enumerate(groups, start=1):
        entry = group["answer"]
        if entry is None:
            skipped.append({"pages": group["pages"], "reason": "no answer page"})
            continue
        entry = {**entry, "question_text": group["stem"]}
        # Sequential numbering, since the source numbers repeat across sessions.
        question = importer._to_backend_question(index, entry)
        if question is None:
            skipped.append({"pages": group["pages"], "reason": "answer page unusable"})
            continue
        question["_source_pages"] = group["pages"]
        question["_figure"] = group["figure"]
        questions.append(question)
    return questions, skipped


def find_dark_figure_bbox(png_bytes: bytes):
    """Locate a clinical figure on a dark-theme page.

    importer._figure_bbox assumes a white page, so on eMedici's dark theme it
    boxes the whole page. Here the background is the page's most common colour
    (not a corner — some screenshots carry a white margin outside the card),
    and a figure is the tallest block of rows dense with non-background pixels.

    Returns (left, top, right, bottom) as page fractions, or None.
    """
    from io import BytesIO
    from PIL import Image

    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    width = 300
    height = max(1, round(img.height * width / img.width))
    px = img.resize((width, height)).load()

    colour_counts = Counter(
        tuple(c // 8 * 8 for c in px[x, y]) for x in range(0, width, 2) for y in range(0, height, 2)
    )
    background = colour_counts.most_common(1)[0][0]
    is_off = [
        [sum(abs(a - b) for a, b in zip(px[x, y], background)) > 45 for x in range(width)]
        for y in range(height)
    ]
    def longest_run(row):
        best_run = run = 0
        for is_pixel_off in row:
            run = run + 1 if is_pixel_off else 0
            best_run = max(best_run, run)
        return best_run

    # A figure row is either mostly non-background (a wide scan, whose grey
    # patches can match the background and break it up) or holds one long
    # unbroken stretch of it (a photo narrower than the page). A line of text
    # is neither: it is broken up by the gaps between words.
    is_dense = [sum(row) / width > 0.45 or longest_run(row) / width > 0.2 for row in is_off]

    min_run = max(8, int(height * 0.06))
    best, start = None, None
    for y, dense in enumerate(is_dense + [False]):
        if dense and start is None:
            start = y
        elif not dense and start is not None:
            if y - start >= min_run and (best is None or y - start > best[1] - best[0]):
                best = (start, y)
            start = None
    if best is None:
        return None

    top, bottom = best
    outside = [y for y in range(height) if not top <= y < bottom] or [0]

    def off_share(x, rows):
        return sum(is_off[y][x] for y in rows) / len(rows)

    # A column that is off-background above and below the figure too is page
    # margin, not figure.
    columns = [
        x for x in range(width)
        if off_share(x, range(top, bottom)) > 0.3 and off_share(x, outside) < 0.6
    ]
    if not columns:
        return None
    return (columns[0] / width, top / height, (columns[-1] + 1) / width, bottom / height)


def crop_figure(pdf_path: str, figure_page: dict):
    """PNG of the figure on that page, or None when it can't be located."""
    page_number = figure_page["_page"]
    full = importer._render_page_png(pdf_path, page_number, 150)
    box = find_dark_figure_bbox(full) or importer._clamp_region(figure_page.get("image_region"), 0.03)
    if not box:
        return None
    return importer._render_page_region_png(pdf_path, page_number, box, 300, False)


async def upload_figure(http, sem, pdf_path: str, figure_page: dict):
    """Store the figure via the target backend. Returns its path, or None — a
    failed figure leaves that question text-only rather than failing the push."""
    async with sem:
        try:
            png = await asyncio.to_thread(crop_figure, pdf_path, figure_page)
            if not png:
                print(f"  figure on page {figure_page['_page']}: not located")
                return None
            resp = await http.post(
                f"{importer.BACKEND_URL}/api/admin/import-batches/images",
                files={"image": (f"q-p{figure_page['_page']}.png", png, "image/png")},
                headers={"Authorization": f"Bearer {importer.ADMIN_TOKEN}"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("path")
        except Exception as e:  # noqa: BLE001
            print(f"  figure on page {figure_page['_page']} failed: {e}")
            return None


async def push(pdf_path: str, cache_path: str, title: str, is_dry_run: bool):
    cache = load_cache(cache_path)
    if not cache:
        sys.exit(f"No cached pages in {cache_path} — run extract first")

    groups = merge_pages(cache)
    questions, skipped = build_questions(groups)
    with_images = [q for q in questions if q["_figure"]]
    no_subject = [q["question_number"] for q in questions if not q.get("subject")]

    subjects = {}
    for q in questions:
        subjects[q.get("subject")] = subjects.get(q.get("subject"), 0) + 1
    print(f"{len(questions)} questions, {len(with_images)} with figures, {len(skipped)} skipped")
    print("subjects:", json.dumps(subjects, indent=1))
    for s in skipped:
        print("  skipped:", s)
    if no_subject:
        print("  no subject:", no_subject)

    if is_dry_run:
        base = cache_path.replace(".json", "")
        with open(base + ".questions.json", "w") as f:
            json.dump(questions, f, indent=1)
        figures_dir = base + "-figures"
        os.makedirs(figures_dir, exist_ok=True)
        for q in with_images:
            png = crop_figure(pdf_path, q["_figure"])
            if png:
                with open(os.path.join(figures_dir, f"q{q['question_number']}.png"), "wb") as f:
                    f.write(png)
        print(f"dry run — questions in {base}.questions.json, figures in {figures_dir}/, "
              f"nothing sent to {importer.BACKEND_URL}")
        return

    print(f"pushing to {importer.BACKEND_URL}")
    async with httpx.AsyncClient() as http:
        batch_id = await importer.create_batch(http, title)
        print(f"created batch {batch_id}")

        upload_sem = asyncio.Semaphore(max(1, importer.QBANK_CONCURRENCY))
        paths = await asyncio.gather(*(
            upload_figure(http, upload_sem, pdf_path, q["_figure"]) for q in with_images
        ))
        for q, path in zip(with_images, paths):
            if path:
                q["images"] = [path]
                q["question_type"] = "image_based"
        print(f"uploaded {sum(1 for p in paths if p)}/{len(with_images)} figures")

        for q in questions:
            for key in [k for k in q if k.startswith("_")]:
                del q[key]

        logs = [f"Skipped (pages {s['pages']}): {s['reason']}" for s in skipped]
        resp = await http.post(
            f"{importer.BACKEND_URL}/api/admin/import-batches/{batch_id}/receive",
            json={
                "status": "success",
                "total_questions": len(questions) + len(skipped),
                "questions": questions,
                "logs": logs,
            },
            headers={"Authorization": f"Bearer {importer.ADMIN_TOKEN}"},
            timeout=300,
        )
        resp.raise_for_status()
        print(f"batch {batch_id}: {len(questions)} questions received")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    is_dry_run = "--dry-run" in sys.argv
    if len(args) == 3 and args[0] == "extract":
        asyncio.run(extract(args[1], args[2]))
    elif len(args) == 4 and args[0] == "push":
        asyncio.run(push(args[1], args[2], args[3], is_dry_run))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
