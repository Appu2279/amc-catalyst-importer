"""
Read eMedici screenshot pages with local OCR (Tesseract) instead of Claude, and
write the same page cache `qbank-pdf-to-batch.py push` reads.

    venv/bin/python scripts/qbank-ocr-to-cache.py <pages-dir> <cache.json> [--ocr-dir=DIR] [--jobs=8]

<pages-dir> holds p1.png ... pN.png rendered at 910px wide (the size the earlier
batches used). Tesseract's word boxes are cached as TSV in --ocr-dir (default
<pages-dir>-ocr) so the parser can be re-run without re-reading pages.

Free to run. What OCR cannot decide is left for a cheap text-only pass:
subject and topic are null here.

How a page is read (eMedici's layout is fixed, so this is rule-based):
  - "Question N" at the top gives the number.
  - The stem is every line above "Choose the single best answer."; its last
    paragraph is the lead-in.
  - Options are the indented lines below it. An answered page shows a "NN%"
    beside each option; the correct one sits on eMedici's green highlight,
    found from pixels, not from text.
  - Discussion paragraphs run from below the options to "REFERENCES". Each goes
    to the options it names; anything general goes to the correct option.
  - Key points are the orange lines, found from text colour.
  - A figure is announced in the stem with an "(Image)" link and sits below that
    paragraph; its box is found from pixels.
"""
import csv
import io
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

GREEN = (217, 245, 238)          # eMedici's correct-answer highlight
# Loose on purpose: this faint line OCRs partly ("ose the sing t answer").
CHOOSE_RE = re.compile(r"choose\s+the|the\s+sing\w*\s+best|single\s+best|best\s+answer|sing\w*\s+\w?\s*answer", re.I)
QUESTION_RE = re.compile(r"\bQuestion\s+(\d{1,3})\b")
PERCENT_RE = re.compile(r"^\d{1,3}%$")
IMAGE_LINK_RE = re.compile(r"\(\s*images?\b|\bimages?\s*\d|\(\s*image", re.I)
BUTTON_RE = re.compile(r"check\s*answer|skip\s*question", re.I)
STOP_RE = re.compile(r"^(REFERENCES|Next Question|AUTHORED BY|LEAVE FEEDBACK|REVIEWED BY)\b", re.I)


# ── OCR ───────────────────────────────────────────────────────────────────────

def ocr_tsv(png_path: str, tsv_path: str) -> str:
    if not os.path.exists(tsv_path):
        # Tried binarising first (coloured text to black): more answer pages
        # detected, but key points and explanations read worse. Colour wins.
        out = subprocess.run(
            ["tesseract", png_path, "-", "--psm", "3", "tsv"],
            capture_output=True, text=True, check=True,
        ).stdout
        with open(tsv_path, "w") as f:
            f.write(out)
    with open(tsv_path) as f:
        return f.read()


def is_icon(word) -> bool:
    return word.get("icon", False)


def mark_icons(line, margin):
    """Mark the tick/cross/radio icons left of an option's text, then recompute
    the line's box from real words (an icon is taller than the text).

    Screenshots are shifted sideways from page to page, so positions are
    relative to the page's text margin. Icons sit around the margin and the
    option text starts ~55px in; a short real word at the margin ("The") runs
    straight on into the next word instead."""
    words = line["words"]
    text_start = next((w["x0"] for w in words if w["x0"] >= margin + 40), None)
    for w in words:
        w["icon"] = bool(ARTIFACT_RE.match(w["text"])) or (
            text_start is not None and w["x1"] < margin + 40 and len(w["text"]) <= 4
            and all(o["x0"] >= text_start or o["x1"] < margin + 40 for o in words))
    # Junk OCR'd from a line's own glyphs is not an icon unless it is left of the text.
    real = [w for w in words if not w["icon"]] or words
    line["y0"], line["y1"] = min(w["y0"] for w in real), max(w["y1"] for w in real)


def read_lines(tsv: str) -> list:
    """Tesseract words grouped into lines, each with word boxes, sorted top-down."""
    lines = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row.get("level") != "5" or not (row.get("text") or "").strip():
            continue
        key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        x, y, w, h = (int(row[k]) for k in ("left", "top", "width", "height"))
        lines.setdefault(key, []).append({"text": row["text"], "x0": x, "y0": y, "x1": x + w, "y1": y + h,
                                          "conf": float(row["conf"])})
    out = []
    for words in lines.values():
        words.sort(key=lambda w: w["x0"])
        out.append({
            "words": words,
            "x0": min(w["x0"] for w in words), "x1": max(w["x1"] for w in words),
            "y0": min(w["y0"] for w in words), "y1": max(w["y1"] for w in words),
        })
    out.sort(key=lambda l: (l["y0"], l["x0"]))
    return out


def content_x0(line) -> int:
    """Left edge of the first real word — the tick/cross icons left of an option
    are OCR'd as junk ("©") and must not count as the line's indent."""
    for w in line["words"]:
        if not is_icon(w):
            return w["x0"]
    return line["x0"]


def line_text(line, min_x=None, max_x=None) -> str:
    words = [w["text"] for w in line["words"]
             if (min_x is None or w["x0"] >= min_x) and (max_x is None or w["x0"] < max_x)]
    return " ".join(words).strip()


# ── Pixels ────────────────────────────────────────────────────────────────────

def near(px, colour, tol=(12, 10, 12)):
    return all(abs(a - b) < t for a, b, t in zip(px, colour, tol))


def green_count(img, box) -> int:
    x0, y0, x1, y1 = (int(v) for v in box)
    crop = img.crop((x0, y0, x1, y1)).resize((max(1, (x1 - x0) // 3), max(1, (y1 - y0) // 3)))
    return sum(1 for px in crop.getdata() if near(px, GREEN))


def text_colour(img, line):
    """Mean colour of the ink in a line — pixels clearly darker than the page."""
    crop = img.crop((line["x0"], line["y0"], line["x1"], line["y1"]))
    ink = [px for px in crop.getdata() if sum(px) < 600]
    if not ink:
        return (255, 255, 255)
    return tuple(sum(c[i] for c in ink) / len(ink) for i in range(3))


def is_orange(colour) -> bool:
    r, g, b = colour
    return r > 170 and 80 < g < 190 and b < 110 and r - b > 100


def is_background(px) -> bool:
    # White card, or the pale blue-grey page around it.
    return min(px) > 236 or (abs(px[0] - 240) < 10 and abs(px[1] - 247) < 8 and abs(px[2] - 249) < 8)


def content_box(img, top, bottom, left, right):
    """Tight box around non-background pixels in a band, or None."""
    top, bottom = int(top), int(bottom)
    if bottom - top < 20:
        return None
    band = img.crop((int(left), top, int(right), bottom))
    w, h = band.size
    px = band.load()
    rows = [y for y in range(0, h, 2) if sum(1 for x in range(0, w, 4) if not is_background(px[x, y])) > 3]
    if not rows:
        return None
    y_a, y_b = rows[0], rows[-1]
    cols = [x for x in range(0, w, 2)
            if sum(1 for y in range(y_a, y_b + 1, 4) if not is_background(px[x, y])) > 3]
    if not cols:
        return None
    return (left + cols[0], top + y_a, left + cols[-1], top + y_b)


# ── Text helpers ──────────────────────────────────────────────────────────────

def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def join_lines(texts: list) -> str:
    out = ""
    for t in texts:
        if not out:
            out = t
        elif out.endswith("-") and not out.endswith(" -"):
            out += t            # hyphenated word split across lines
        else:
            out += " " + t
    return out.strip()


def paragraphs(lines: list, gap_factor=1.3) -> list:
    """Group consecutive lines into paragraphs by vertical gap."""
    groups = []
    for line in lines:
        h = max(12, line["y1"] - line["y0"])
        if groups and line["y0"] - groups[-1][-1]["y1"] < gap_factor * h:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


ARTIFACT_RE = re.compile(r"^[^A-Za-z0-9]+$|^[^A-Za-z0-9]*[A-Za-z]{1,2}[^A-Za-z0-9]+$")


def strip_bullet(text: str) -> str:
    """Drop a leading icon OCR'd as junk ("“@", "'tS-", "*", "©") from a line."""
    parts = text.split(" ", 1)
    if len(parts) == 2 and ARTIFACT_RE.match(parts[0]):
        return parts[1].strip()
    return text.strip()


# ── Page parser ───────────────────────────────────────────────────────────────

def parse_page(png_path: str, tsv: str, page_number: int) -> dict:
    img = Image.open(png_path).convert("RGB")
    W, H = img.size
    lines = read_lines(tsv)
    # The page's text margin: the most common left edge (stem and discussion lines).
    from collections import Counter
    edges = Counter(l["x0"] // 6 * 6 for l in lines if l["x0"] < 0.2 * W)
    margin = (edges.most_common(1)[0][0] + 3) if edges else 42
    for line in lines:
        mark_icons(line, margin)
    lines = [l for l in lines if any(not w["icon"] for w in l["words"])]
    lines.sort(key=lambda l: (l["y0"], l["x0"]))
    # Tesseract sometimes splits one visual row (an option and its "NN%") into
    # two lines; join lines that overlap vertically.
    merged = []
    for line in lines:
        prev = merged[-1] if merged else None
        if prev is not None:
            overlap = min(prev["y1"], line["y1"]) - max(prev["y0"], line["y0"])
            if overlap > 0.5 * min(prev["y1"] - prev["y0"], line["y1"] - line["y0"]) and \
                    (line["x0"] > prev["x1"] or prev["x0"] > line["x1"]):
                prev["words"] = sorted(prev["words"] + line["words"], key=lambda w: w["x0"])
                prev["x0"], prev["x1"] = min(prev["x0"], line["x0"]), max(prev["x1"], line["x1"])
                continue
        merged.append(line)
    lines = merged
    option_x = margin + 35        # options are indented past the radio / tick icon
    percent_x = 0.85 * W

    page = {"_page": page_number, "has_answer": False, "question_number": None, "question_text": "",
            "subject": None, "topic": None, "options": [], "key_points": [],
            "has_image": False, "image_region": None}

    # Header ("Question N  Share  Save") — only near the top of the screenshot.
    body = []
    for line in lines:
        m = QUESTION_RE.search(line_text(line))
        if m and line["y0"] < 0.12 * H and page["question_number"] is None:
            page["question_number"] = int(m.group(1))
            continue
        body.append(line)

    # The "Choose" line is short; the same words can turn up inside the discussion.
    choose = next((i for i, l in enumerate(body)
                   if CHOOSE_RE.search(line_text(l)) and len(line_text(l).split()) <= 7), None)
    if choose is None:
        # OCR sometimes misses the faint "Choose the single best answer." line.
        # Fall back to the lead-in: a margin line ending in "?" whose next line is indented.
        for k in range(len(body) - 1):
            if (line_text(body[k]).rstrip().endswith("?") and content_x0(body[k]) < option_x
                    and content_x0(body[k + 1]) >= option_x):
                choose = k          # treated like the "Choose" line, but it is stem text
                body = body[:k + 1] + [{"words": [], "x0": 0, "x1": 0,
                                        "y0": body[k]["y1"], "y1": body[k]["y1"]}] + body[k + 1:]
                choose = k + 1
                break
    stem_lines = body[:choose] if choose is not None else []
    rest = body[choose + 1:] if choose is not None else body

    # ── Options: indented rows straight after the "Choose" line ──
    option_rows, i = [], 0
    while i < len(rest):
        line = rest[i]
        text = line_text(line)
        if content_x0(line) < option_x or STOP_RE.match(text) or BUTTON_RE.search(text):
            break
        # The "Check Answer / Skip Question" row sits well below the last option;
        # its button text OCRs badly, so stop on the gap rather than the words.
        if option_rows and line["y0"] - option_rows[-1]["y1"] > 2.6 * max(18, line["y1"] - line["y0"]):
            break
        option_rows.append(line)
        i += 1
    discussion_lines = rest[i:]

    # A scrolled answer page can start at the options with no "Choose" line.
    if choose is None:
        # Option rows end in their "NN%" and sit together; a "1%" inside later
        # text (a key point) must not stretch the options down the page.
        lead = []
        for l in body:
            last = l["words"][-1]
            if PERCENT_RE.match(last["text"]) and last["x0"] > percent_x:
                if lead and l["y0"] - lead[-1]["y1"] > 120:
                    break
                lead.append(l)
        if len(lead) >= 3:
            start = body.index(lead[0])
            stem_lines = body[:start]
            option_rows = [l for l in body[start:] if content_x0(l) >= option_x and l["y0"] <= lead[-1]["y1"] + 40]
            discussion_lines = [l for l in body[start:] if l["y0"] > lead[-1]["y1"] + 10 and l not in option_rows]
        else:
            stem_lines, option_rows, discussion_lines = body, [], []

    # Wrapped option text continues on the next line with a small gap.
    def has_pct(line):
        return any(w["x0"] > percent_x for w in line["words"])

    if sum(has_pct(l) for l in option_rows) >= 3:
        groups = []
        for line in option_rows:
            if has_pct(line) or not groups:
                groups.append([line])
            elif line["y0"] - groups[-1][-1]["y1"] < 1.3 * max(18, line["y1"] - line["y0"]):
                groups[-1].append(line)     # wrapped text of the option above
            else:
                break
    else:
        groups = paragraphs(option_rows, gap_factor=1.2)

    options = []
    for group in groups:
        texts = [line_text(l, min_x=option_x - 10, max_x=percent_x) for l in group]
        text = join_lines([t for t in texts if t])
        text = re.sub(r"\s+\d{1,3}%$", "", text).strip()
        has_percent = any(w["x0"] > percent_x for l in group for w in l["words"])
        if text:
            options.append({"text": strip_bullet(text), "box": (option_x - 40, group[0]["y0"] - 12,
                                                                0.8 * W, group[-1]["y1"] + 12),
                            "has_percent": has_percent})
    greens = [green_count(img, o["box"]) for o in options]
    best = max(range(len(options)), key=lambda k: greens[k]) if options else None
    correct = best if best is not None and greens[best] > 40 else None
    page["has_answer"] = correct is not None and (sum(o["has_percent"] for o in options) >= 2 or greens[best] > 150)
    if not page["has_answer"]:
        correct = None

    # ── Stem and figure ──
    # A figure is a tall band between lines of stem text that holds non-background
    # pixels. Text OCR'd inside a photo is junk (low confidence), so only
    # confident lines bound the gaps.
    def confident(line):
        words = [w for w in line["words"] if not w["icon"]]
        return words and sum(w["conf"] for w in words) / len(words) >= 70 and len(line_text(line)) >= 3

    if not page["has_answer"] and stem_lines:
        anchors = [l for l in stem_lines if confident(l)]
        bottom_edge = rest[0]["y0"] if rest else H
        bands = [(a["y1"], b["y0"]) for a, b in zip(anchors, anchors[1:])]
        if anchors:
            bands.append((anchors[-1]["y1"], bottom_edge))
        best_box = None
        for top, bottom in bands:
            if bottom - top < 0.05 * H:
                continue
            box = content_box(img, top + 4, bottom - 4, 0.03 * W, 0.97 * W)
            # Not a figure if it holds the lead-in or the "Choose" line — that is
            # a page whose short options ("8", "9") left a text-only gap.
            if box and any(box[1] - 2 <= l["y0"] and l["y1"] <= box[3] + 2 and
                           (CHOOSE_RE.search(line_text(l)) or line_text(l).rstrip().endswith("?"))
                           for l in body):
                box = None
            if box and (box[3] - box[1]) > 0.04 * H and (best_box is None or
                                                         (box[3] - box[1]) > (best_box[3] - best_box[1])):
                best_box = box
        if best_box:
            page["has_image"] = True
            page["image_region"] = [round(best_box[0] / W, 4), round(best_box[1] / H, 4),
                                    round(best_box[2] / W, 4), round(best_box[3] / H, 4)]
            # Lines inside the figure (photo junk, or a lab table image) are not stem text.
            stem_lines = [l for l in stem_lines if not (l["y0"] >= best_box[1] - 2 and l["y1"] <= best_box[3] + 2)]
    stem_paras = paragraphs(stem_lines)
    page["question_text"] = "\n\n".join(join_lines([line_text(l) for l in p]) for p in stem_paras).strip()

    # ── Discussion and key points ──
    disc, keys = [], []
    for line in discussion_lines:
        text = line_text(line)
        if STOP_RE.match(text):
            break
        (keys if is_orange(text_colour(img, line)) else disc).append(line)

    # A key point starts at its light-bulb icon (OCR'd as junk, or not at all —
    # then the wider gap between points still separates them).
    points = []
    for line in keys:
        starts = any(w["icon"] for w in line["words"]) or ARTIFACT_RE.match(line["words"][0]["text"])
        h = max(12, line["y1"] - line["y0"])
        if not points or starts or line["y0"] - points[-1][-1]["y1"] > 0.9 * h:
            points.append([line])
        else:
            points[-1].append(line)
    page["key_points"] = [strip_bullet(join_lines([line_text(l, min_x=None) for l in p])) for p in points]
    disc_paras = [strip_bullet(join_lines([line_text(l) for l in p])) for p in paragraphs(disc)]

    page["options"] = [{"text": o["text"], "is_correct": page["has_answer"] and k == correct, "explanation": None}
                       for k, o in enumerate(options)]
    page["_paras"] = disc_paras if page["has_answer"] else []
    return page


def assign_explanations(page):
    """Each discussion paragraph goes to the options it names; the rest to the correct one."""
    paras = page.pop("_paras", [])
    if not page["has_answer"]:
        return
    correct = next((k for k, o in enumerate(page["options"]) if o["is_correct"]), None)
    keys_norm = [norm(o["text"]) for o in page["options"]]
    found = [[] for _ in page["options"]]
    for para in paras:
        p = norm(para)
        named = [k for k, key in enumerate(keys_norm)
                 if key and (key in p or (len(key.split()) > 6 and " ".join(key.split()[:6]) in p))]
        for k in (named or [correct]):
            found[k].append(para)
    for k, o in enumerate(page["options"]):
        o["explanation"] = "\n\n".join(found[k]) or None


def use_question_page_text(cache):
    """On an answer page the correct option is green on green and OCRs badly
    ("Epiaiaymitis"). Take option text from the matching question page, which
    is black on white, when the two pages clearly show the same options."""
    from difflib import SequenceMatcher
    for n in sorted(int(k) for k in cache):
        page = cache[str(n)]
        if not page["has_answer"]:
            continue
        for back in (1, 2, 3):
            q = cache.get(str(n - back))
            if not q or q["has_answer"] or len(q["options"]) != len(page["options"]):
                continue
            sims = [SequenceMatcher(None, norm(a["text"]), norm(b["text"])).ratio()
                    for a, b in zip(q["options"], page["options"])]
            if sum(sims) / len(sims) > 0.75 and min(sims) > 0.4:
                for a, b in zip(q["options"], page["options"]):
                    b["text"] = a["text"]
                break


# ── Unit repair ───────────────────────────────────────────────────────────────
# Tesseract flattens superscripts and drops µ. These rules undo the forms it
# actually produced on eMedici pages (counted on the 1501-1750 batch); anything
# ambiguous is left alone rather than guessed.
SUPER = r"(?:\^|\*|°|“|%|\?)"


def _ten_power(m):
    before = m.string[max(0, m.start() - 45):m.start()].lower()
    exp = m.group(1) or ("12" if re.search(r"red (?:cell|blood)|rbc|erythrocyte", before) else "9")
    return f"x10^{exp}/L"


UNIT_RULES = [
    # x10^9/L in all its OCR forms: x109/L, x 10*9/L, x 10°/L, x10? /L, x1012/L
    (re.compile(rf"[x×]\s?10\s?{SUPER}?\s?(12|9)?\s?{SUPER}?\s?/\s?L\b"), _ten_power),
    # µmol/L: "u", "j", "ji", "y", "[" are all a lost µ; bare "mol/L" is never a clinical unit.
    (re.compile(r"\b(?:u|ji|j|y)mol/L\b|\[mol/L\b"), "µmol/L"),
    (re.compile(r"(?<=\d )mol/L\b"), "µmol/L"),
    (re.compile(r"((?:creatinine|bilirubin|urate|uric acid|\(cr\))[^;\n]{0,30}?\d\s?)pmol/L", re.I), r"\1µmol/L"),
    (re.compile(r"\b(?:ji|j|u)g/L\b"), "µg/L"),
    (re.compile(r"kg/m(?:\?|®|2)?(?=[\s.,;)]|$)"), "kg/m2"),
    (re.compile(r"\bPaCO[,.]?(?=[\s:])"), "PaCO2"),
    (re.compile(r"\bPaO[,.]?(?=[\s:])"), "PaO2"),
    (re.compile(r"\bSpO[,.]?(?=[\s:])"), "SpO2"),
    (re.compile(r"\bHbA(?:\\c|lc|1C|1e)\b"), "HbA1c"),
    (re.compile(r"((?:e?GFR)[^()]{0,25}\()2(\d{2})\)"), r"\1>\2)"),
    # Roman numerals after a grading word: "Class |" -> "Class I", "Class I!" -> "Class II".
    (re.compile(r"\b(Class|Grade|Stage|Type|Phase|Level|Degree|Mobitz|Tier|Category|Lead)\s+"
                r"([|Il1!]{1,3})(?=$|[\s.,;:)])"), lambda m: f"{m.group(1)} {_roman(m.group(2))}"),
    (re.compile(r"\b(Mirena)\s+\|UD\b"), r"\1 IUD"),
]


def _roman(token):
    # Only I-strokes are ambiguous; "1" stays a digit (Type 1 diabetes).
    return token if token == "1" else "I" * len(token)


def fix_units(text):
    if not text:
        return text
    for pattern, repl in UNIT_RULES:
        text = pattern.sub(repl, text)
    return text


def fix_page_units(page):
    page["question_text"] = fix_units(page["question_text"])
    page["key_points"] = [fix_units(k) for k in page["key_points"]]
    for o in page["options"]:
        o["text"] = fix_units(o["text"])
        o["explanation"] = fix_units(o["explanation"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) != 2:
        sys.exit(__doc__)
    pages_dir, cache_path = args
    ocr_dir = opts.get("ocr-dir", pages_dir.rstrip("/") + "-ocr")
    jobs = int(opts.get("jobs", "8"))
    os.makedirs(ocr_dir, exist_ok=True)

    numbers = sorted(int(m.group(1)) for f in os.listdir(pages_dir) if (m := re.match(r"p(\d+)\.png$", f)))

    def work(n):
        png = os.path.join(pages_dir, f"p{n}.png")
        tsv = ocr_tsv(png, os.path.join(ocr_dir, f"p{n}.tsv"))
        return n, parse_page(png, tsv, n)

    cache = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for done, (n, page) in enumerate(pool.map(work, numbers), start=1):
            cache[str(n)] = page
            if done % 50 == 0:
                print(f"  {done}/{len(numbers)}")

    use_question_page_text(cache)
    for page in cache.values():
        assign_explanations(page)
        fix_page_units(page)

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=1)
    answered = sum(1 for p in cache.values() if p["has_answer"])
    print(f"{len(cache)} pages, {answered} answered, {sum(p['has_image'] for p in cache.values())} figures -> {cache_path}")


if __name__ == "__main__":
    main()
