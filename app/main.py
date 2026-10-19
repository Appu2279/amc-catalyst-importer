from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import tempfile
import os
import uuid
import json
import base64
import asyncio
import pdfplumber
import fitz  # PyMuPDF
import httpx
import re
from collections import deque, Counter

import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# Load .env from the project root (one level above app/)
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_env_path)

# ==========================================
# CONFIG
# ==========================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000").strip()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

# ── QBank extraction (Claude vision) ──────────────────────────────────────────
# Only the /import/qbank endpoint uses these. The recall /import path does not
# touch Claude, so the service still runs fine with no key set.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
QBANK_MODEL       = os.getenv("QBANK_MODEL", "claude-haiku-4-5").strip()
# How many pages to read from Claude at once. 5 is comfortably under the default
# rate limits and keeps a 400-page PDF to a few minutes.
QBANK_CONCURRENCY = int(os.getenv("QBANK_CONCURRENCY", "5"))
# Render DPI for each page image. 130 keeps the long edge near Anthropic's
# 1568px downscale threshold, so higher values cost more tokens for no gain.
QBANK_DPI         = int(os.getenv("QBANK_DPI", "130"))

_cloud_name   = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
_cloud_key    = os.getenv("CLOUDINARY_API_KEY",    "").strip()
_cloud_secret = os.getenv("CLOUDINARY_API_SECRET", "").strip()

cloudinary.config(
    cloud_name = _cloud_name,
    api_key    = _cloud_key,
    api_secret = _cloud_secret,
    secure     = True,
)

# ==========================================
# APP
# ==========================================
app = FastAPI(title="AMC Catalyst Import Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# HEALTH CHECK
# ==========================================
@app.get("/")
async def root():
    return {"status": "success", "message": "AMC Catalyst Import Service Running"}


# ==========================================
# PDF HELPERS
# ==========================================
def clean_text(text):
    if not text:
        return text
    text = text.replace('\n', ' ')
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# Options run A..F. The original recall format only ever used A-D, but the
# newer prose format uses five, so nothing downstream may assume four.
OPTION_LETTERS = "ABCDEF"

# A running page footer/header some source exports print on every page, with
# no space between the page number and the text that follows it — e.g.
# "Page 4Dr Solosailor's AMC Catalyst Questions 151 to 179". Because it's
# glued rather than separated by whitespace, it lands stuck onto the end of
# whatever real content happened to be last on that page (an option's text, an
# explanation) instead of standing apart where a parser boundary could skip
# it. Stripped per-page, right at extraction, since by the time pages are
# concatenated there's no reliable boundary left to find it at.
_PAGE_FOOTER_RE = re.compile(r"Page\s*\d+[A-Z][^\n]{0,80}?(?:Questions|Answers)\s+\d+\s+to\s+\d+\s*")


def _strip_page_footer(page_text):
    return _PAGE_FOOTER_RE.sub(' ', page_text) if page_text else page_text


def extract_pdf_text_pages(pdf_path):
    """Same text as extract_pdf_text, plus where each page starts in it.

    The prose format has no [Page Number] marker, so the page a question sits
    on has to be derived from its offset in the text — and that page number is
    what attaches the extracted images to the right question.
    """
    text = ""
    page_starts = []
    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages):
            page_starts.append((len(text), index + 1))
            page_text = _strip_page_footer(page.extract_text())
            if page_text:
                text += page_text + "\n"
    return text, page_starts


def page_for_offset(page_starts, offset):
    """Which 1-based page an offset in the concatenated text falls on."""
    page = None
    for start, number in page_starts or []:
        if start <= offset:
            page = number
        else:
            break
    return page


def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = _strip_page_footer(page.extract_text())
            if page_text:
                text += page_text + "\n"
    return text


def upload_to_cloudinary(local_path: str, folder: str, public_id: str) -> str:
    """Upload a local image file to Cloudinary and return its secure URL."""
    result = cloudinary.uploader.upload(
        local_path,
        folder=folder,
        public_id=public_id,
        resource_type="image",
        overwrite=True,
    )
    return result["secure_url"]


def extract_pdf_images(pdf_path, job_id, prefix="img"):
    """
    Uses pdfplumber to detect image bounding boxes, renders each region via
    PyMuPDF at 2x resolution, uploads every image to Cloudinary, and returns
    {page_number: [cloudinary_url]}.

    Temp PNG files are written to a system temp dir and deleted after upload.
    """
    images_by_page = {}
    cloudinary_folder = f"amc-catalyst/imports/{job_id}"

    fitz_doc = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, plumber_page in enumerate(pdf.pages):
            page_num = page_idx + 1
            plumber_images = plumber_page.images

            if not plumber_images:
                continue

            fitz_page = fitz_doc[page_idx]
            page_image_urls = []

            for img_idx, img in enumerate(plumber_images):
                rect = fitz.Rect(img["x0"], img["top"], img["x1"], img["bottom"])
                pix  = fitz_page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)

                filename  = f"{prefix}_page{page_num}_img{img_idx}"
                tmp_path  = os.path.join(tempfile.gettempdir(), f"{job_id}_{filename}.png")

                try:
                    pix.save(tmp_path)
                    url = upload_to_cloudinary(tmp_path, cloudinary_folder, filename)
                    page_image_urls.append(url)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            if page_image_urls:
                images_by_page[page_num] = page_image_urls

    fitz_doc.close()
    return images_by_page


_ANSWER_PAGE_MAP_PATTERNS = (
    re.compile(r'Question Number:\s*(\d+)', re.IGNORECASE),
    re.compile(r'(?m)^Q(\d+)\s*[—–-]\s*Correct:', re.IGNORECASE),
    re.compile(r'(?m)^Q(\d+)\.\s*Answer:\s*[A-F]\b', re.IGNORECASE),
)


def get_answer_page_map(pdf_path):
    """Returns {question_number: page_number} by scanning each page of the answers PDF.

    Tries every known answer-header shape (labelled and bracketed formats) since
    this runs before either parser has identified which one the PDF actually is.
    """
    page_map = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            for pattern in _ANSWER_PAGE_MAP_PATTERNS:
                for match in pattern.finditer(page_text):
                    page_map[int(match.group(1))] = page_idx + 1
    return page_map


# ==========================================
# PARSERS
# ==========================================
def _parse_questions_labelled(text):
    questions = []
    qn_pattern = re.compile(r'Question Number:\s*(\d+)', re.IGNORECASE)
    qn_matches = list(qn_pattern.finditer(text))

    for idx, qn_match in enumerate(qn_matches):
        question_number = int(qn_match.group(1))

        after_start = qn_match.end()
        after_end   = qn_matches[idx + 1].start() if idx + 1 < len(qn_matches) else len(text)
        after_block = text[after_start:after_end]

        before_start = qn_matches[idx - 1].end() if idx > 0 else 0
        before_block = text[before_start:qn_match.start()]

        categories = re.findall(r'\[([^\]:]+)\]', before_block)

        difficulty_match = re.search(r'\[Difficulty:\s*([^\]]+)\]', after_block)
        difficulty = difficulty_match.group(1).strip() if difficulty_match else None

        image_present_match = re.search(r'\[Image Present:\s*([^\]]+)\]', after_block)
        image_present_str   = image_present_match.group(1).strip() if image_present_match else None
        image_present       = image_present_str == "Yes" if image_present_str else False

        image_type_match = re.search(r'\[Image Type:\s*([^\]]+)\]', after_block)
        image_type = image_type_match.group(1).strip() if image_type_match else None

        page_number_match = re.search(r'\[Page Number:\s*([^\]]+)\]', after_block)
        try:
            page_number = int(page_number_match.group(1).strip()) if page_number_match else None
        except ValueError:
            page_number = None

        option_match = re.search(
            r'(?:Options:\s*)?A\.\s*([^\n]+)\nB\.\s*([^\n]+)\nC\.\s*([^\n]+)\nD\.\s*([^\n]+)',
            after_block
        )
        if not option_match:
            continue

        if 'Options:' in after_block:
            q_raw = after_block[:after_block.index('Options:')]
        else:
            q_raw = after_block[:option_match.start()]

        q_raw         = re.sub(r'\[[^\]]+\]\s*\n?', '', q_raw)
        question_text = ' '.join(q_raw.split())

        options = {
            "A": clean_text(option_match.group(1)),
            "B": clean_text(option_match.group(2)),
            "C": clean_text(option_match.group(3)),
            "D": clean_text(option_match.group(4)),
        }
        # Five-option questions exist in newer papers; the A-D block above stays
        # required so this cannot change how an existing paper parses.
        for extra in ("E", "F"):
            extra_match = re.search(rf'(?m)^{extra}\.\s*(.+)$', after_block)
            if extra_match:
                options[extra] = clean_text(extra_match.group(1))

        questions.append({
            "question_number": question_number,
            "subject":         clean_text(categories[0]) if categories else None,
            "topic":           clean_text(categories[1]) if len(categories) > 1 else None,
            "difficulty":      clean_text(difficulty).lower() if difficulty else None,
            "source_type":     "recall",
            "question_type":   "image_based" if image_present else "single_choice",
            "marks":           1,
            "negative_marks":  0,
            "image_present":   image_present,
            "image_type":      clean_text(image_type),
            "page_number":     page_number,
            "question_text":   clean_text(question_text),
            "options": options,
        })

    return questions


def _parse_answers_labelled(text):
    # question_number -> list of answer entries, in document order. A plain
    # dict would silently let a later duplicate number overwrite an earlier
    # one; merge_questions_answers pairs occurrences up in order instead.
    answers    = {}
    qn_pattern = re.compile(r'Question Number:\s*(\d+)', re.IGNORECASE)
    qn_matches = list(qn_pattern.finditer(text))

    for idx, qn_match in enumerate(qn_matches):
        question_number = int(qn_match.group(1))

        after_start = qn_match.end()
        after_end   = qn_matches[idx + 1].start() if idx + 1 < len(qn_matches) else len(text)
        block       = text[after_start:after_end]

        correct_match = re.search(r'Correct Answer:\s*([A-F])\.', block, re.IGNORECASE)
        if not correct_match:
            continue

        answer_letter = correct_match.group(1).upper()

        explanation_match = re.search(
            r'Explanation:\s*(.*?)(?=Option-wise Explanation:|$)',
            block, re.DOTALL | re.IGNORECASE
        )
        explanation = clean_text(explanation_match.group(1)) if explanation_match else None

        option_explanations = {}
        option_wise_match = re.search(r'Option-wise Explanation:(.*)', block, re.DOTALL | re.IGNORECASE)
        if option_wise_match:
            opt_text = option_wise_match.group(1)
            for letter in OPTION_LETTERS:
                opt_match = re.search(rf'{letter}\.\s*(.*?)(?=\n[A-F]\.\s|\Z)', opt_text, re.DOTALL)
                if opt_match:
                    option_explanations[letter] = clean_text(opt_match.group(1))

        answers.setdefault(question_number, []).append({
            "correct_answer":      answer_letter,
            "explanation":         explanation,
            "option_explanations": option_explanations if option_explanations else None,
        })

    return answers


def strip_option_prefix(explanation, option_text):
    if not explanation or not option_text:
        return explanation
    if explanation.lower().startswith(option_text.lower()):
        remainder = explanation[len(option_text):].lstrip(' :-–—')
        if not remainder:
            return explanation
        # Removing the repeated option text leaves a sentence fragment starting
        # mid-thought ("wrong; this refers to..."), so the first letter is
        # restored to upper case.
        return remainder[0].upper() + remainder[1:]
    return explanation


# ── Prose format ─────────────────────────────────────────────────────────────
# A second paper layout, with no bracketed metadata. Papers we've seen mix two
# question-header spellings *within the same document* (a paper stitched
# together from more than one exam session):
#
#   Question 1
#   <stem paragraphs>
#   Which one of the following is the most appropriate management?
#   A. First option B. Second option C. Third option
#
#   Q135. <stem, sometimes preceded by a short title line>
#   A. First option B. Second option C. Third option
#
# and answers, similarly mixing "Answer:"/"Correct answer:", "."/")" after the
# option letter, and the declaration sharing a line with the header or sitting
# on its own line below it:
#
#   Q1 — Answer: C. Third option
#   <explanation paragraphs>
#   ● A. First option — wrong; ...
#
#   Q203. <title line>
#   Correct answer: C. Third option
#   ● A. First option — Incorrect: ...
#
# Note options often run inline and wrap mid-option across lines, so they
# cannot be split on newlines the way the labelled format is.

_PROSE_QUESTION_RE = re.compile(
    r'(?m)^(?:Question\s+(?P<n1>\d+)\s*$'
    r'|Q(?P<n2>\d+)(?:\s*\([^)]*\))?\.\s*(?=\S))'
)

# Header only: "Q135.", "Q1 —", "Q140 (additional stem).". Deliberately loose
# about what follows the number — the correct-answer declaration is found
# separately within the block, since it can share this line or sit on the
# next one.
_PROSE_ANSWER_HEADER_RE = re.compile(r'(?m)^Q(\d+)(?:\s*\([^)]*\))?\s*[.—–-]')

# "Answer: C.", "Correct answer: B —", "Answer: B)" — wherever it falls in the
# block following a header.
_PROSE_ANSWER_DECLARATION_RE = re.compile(
    r'(?:Correct\s+answer|Answer)\s*:\s*([A-F])\s*[.)—–-]', re.IGNORECASE
)


def _split_inline_options(chunk):
    """Separate an inline 'A. x B. y C. z' run from the stem before it.

    Only a run that starts at A and steps forward one letter at a time counts,
    so an 'A.' inside the stem cannot start the options early — the sequence
    has to continue with B for anything to be taken as an option. Accepts
    either 'A.' or 'A)' as the option marker — both appear across papers.
    """
    flat = ' '.join(chunk.split())

    markers = []
    for match in re.finditer(r'(?:(?<=\s)|^)([A-F])[.)]\s', flat):
        if match.group(1) == OPTION_LETTERS[len(markers)]:
            markers.append(match)
            if len(markers) == len(OPTION_LETTERS):
                break

    if len(markers) < 2:
        return flat, {}

    options = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(flat)
        options[marker.group(1)] = clean_text(flat[marker.end():end])

    return clean_text(flat[:markers[0].start()]), options


def _parse_questions_prose(text, page_starts=None):
    questions = []
    matches = list(_PROSE_QUESTION_RE.finditer(text))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)

        question_text, options = _split_inline_options(text[start:end])
        if len(options) < 2 or not question_text:
            continue

        questions.append({
            "question_number": int(match.group('n1') or match.group('n2')),
            # This format carries no subject/topic/difficulty markers. They are
            # left unset rather than guessed, and can be filled in from the
            # admin question editor.
            "subject":         None,
            "topic":           None,
            "difficulty":      None,
            "source_type":     "recall",
            "question_type":   "single_choice",
            "marks":           1,
            "negative_marks":  0,
            "image_present":   False,
            "image_type":      None,
            "page_number":     page_for_offset(page_starts, match.start()),
            "question_text":   question_text,
            "options":         options,
        })

    return questions


def _parse_answers_prose(text):
    # question_number -> list of answer entries, in document order — see the
    # note on _parse_answers_labelled's `answers` for why this isn't a
    # question_number -> entry dict.
    answers = {}
    matches = list(_PROSE_ANSWER_HEADER_RE.finditer(text))

    for index, match in enumerate(matches):
        question_number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]

        declaration = _PROSE_ANSWER_DECLARATION_RE.search(block)
        if not declaration:
            continue

        # Skip past the rest of the declaration's line: it repeats the
        # correct option's text ("Answer: C. Maintain sertraline at 50 mg
        # daily"), which would otherwise be read as the opening words of the
        # explanation.
        line_end = block.find('\n', declaration.end())
        body = block[line_end + 1:] if line_end != -1 else block[declaration.end():]

        # The per-option notes are bulleted; everything before the first bullet
        # is the explanation of the correct answer.
        bullet_split = re.split(r'[●•]', body, maxsplit=1)
        explanation = clean_text(' '.join(bullet_split[0].split()))

        option_explanations = {}
        for letter, exp_text in re.findall(r'[●•]\s*([A-F])[.)]\s*([^●•]+)', body):
            option_explanations[letter] = clean_text(' '.join(exp_text.split()))

        answers.setdefault(question_number, []).append({
            "correct_answer":      declaration.group(1).upper(),
            "explanation":         explanation or None,
            "option_explanations": option_explanations,
        })

    return answers


# ── Bracketed format ─────────────────────────────────────────────────────────
# A third paper layout:
#
#   Q1. [Cardiology] (Medium) A 62-year-old man presents with... Which of the
#   following is the most appropriate immediate next step?
#   A. Administer sublingual glyceryl trinitrate B. Administer intravenous
#   atropine and avoid nitrates C. Proceed directly to primary PCI... D. ... E. ...
#
# and answers, keyed by option WORDING rather than letter:
#
#   Q1 — Correct: A Reference: ... Why A is correct: <explanation>
#   ● B. GTN — drops preload further, can precipitate cardiogenic shock...
#   ● C. ... ● D. ... ● E. ...
#
# The critical trap: the letter this format prints is NOT the same letter as
# the question paper's for the same option — each PDF was generated with its
# own independent shuffle of A-E. Cross-checking option wording between real
# question/answer pairs confirms this (e.g. an option lettered A in the
# question paper turns up lettered B in the answer key, verbatim). So unlike
# the labelled and prose formats, merge_questions_answers cannot pair these by
# letter — _resolve_bracketed_answer matches by wording instead, before the
# normal letter-keyed merge ever runs.

_BRACKETED_QUESTION_RE = re.compile(
    # Not anchored to the start of a line: some source exports glue this
    # header directly onto the end of the previous question's last option
    # with no separating newline or space at all (a missing line break in the
    # source, not a rendering artifact — confirmed by opening the PDF). The
    # "Q<n>. [" shape is distinctive enough that matching it anywhere is safe.
    r'Q(\d+)\.\s*\[([^\]]+)\]\s*'
    # Difficulty is printed either "(Medium)" (captured here) or as one of the
    # dash-separated tags below — but earlier batches print no difficulty at
    # all, so this whole group is optional rather than required.
    r'(?:\(([^)]+)\)\s*)?'
    # Everything after that can be zero or more "— <tag>" segments in any
    # order and combination — difficulty, a setting ("Hospital setting"),
    # "Image-based", seen both alone and stacked ("— Hard — Hospital setting
    # — Image-based"). Rather than model every combination, this just grabs
    # the whole raw run as one span (group 4) so the parser can pull the
    # difficulty/image-based signal out of it afterwards — the earlier
    # per-tag alternation only ever matched ONE such segment and left any
    # more stacked after it sitting unconsumed, leaking onto the front of the
    # stem.
    r'((?:\s*[—–-]\s*[^—–\-\n]+)*)\s*',
    re.IGNORECASE,
)


def _parse_questions_bracketed(text, page_starts=None):
    """Options run inline like the prose format and, in a well-formed source,
    in strict A, B, C... order — so this reuses _split_inline_options rather
    than the answer key's wording-based matching (see the format note above;
    that shuffle is specific to the answer key, not the question paper).

    Some source papers omit the stem's trailing '?' entirely (a lead-in
    question implied rather than printed, e.g. "...no rib fractures are seen.
    A. Insert a left intercostal catheter..."), so the split can't rely on
    locating a question mark — it finds where the options begin instead,
    which works whether or not the stem happens to end in one.
    """
    questions = []
    matches = list(_BRACKETED_QUESTION_RE.finditer(text))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = ' '.join(text[start:end].split())

        stem, options = _split_inline_options(block)
        if len(options) < 2 or not stem:
            continue

        # A stray paragraph of leftover authoring notes occasionally lands on
        # the last option — see _trim_runaway_tail.
        letters = list(options)
        texts = _trim_runaway_tail([options[letter] for letter in letters])
        options = dict(zip(letters, texts))

        subject_raw = match.group(2)
        tag_suffix  = match.group(4) or ""
        image_present = (
            bool(re.search(r'image[\s-]*based', subject_raw, re.IGNORECASE))
            or bool(re.search(r'image[\s-]*based', tag_suffix, re.IGNORECASE))
            or bool(re.search(r'\(Image\s*\d+\)', stem, re.IGNORECASE))
        )
        subject_clean = re.sub(r'\s*[—,-]\s*image[\s-]*based\s*$', '', subject_raw, flags=re.IGNORECASE).strip()
        # A compound label ("Public Health / Immunisation", "Ethics / Child
        # Protection") is a subject/topic pair, not one subject — split it the
        # same way the labelled format's two brackets already do.
        if '/' in subject_clean:
            subject_part, topic_part = subject_clean.split('/', 1)
            subject, topic = subject_part.strip(), topic_part.strip() or None
        else:
            subject, topic = subject_clean, None

        difficulty_match = re.search(r'\b(Easy|Medium|Hard)\b', tag_suffix, re.IGNORECASE)
        difficulty_raw = match.group(3) or (difficulty_match.group(1) if difficulty_match else None)

        questions.append({
            "question_number": int(match.group(1)),
            "subject":         clean_text(subject),
            "topic":           clean_text(topic),
            "difficulty":      clean_text(difficulty_raw).lower() if difficulty_raw else None,
            "source_type":     "recall",
            "question_type":   "image_based" if image_present else "single_choice",
            "marks":           1,
            "negative_marks":  0,
            "image_present":   image_present,
            "image_type":      None,
            "page_number":     page_for_offset(page_starts, match.start()),
            "question_text":   stem,
            "options":         options,
        })

    return questions


_BRACKETED_ANSWER_HEADER_RE = re.compile(r'(?m)^Q(\d+)\s*[—–-]\s*Correct:\s*[A-F]\b')
_BRACKETED_WHY_RE           = re.compile(r'Why\s+[A-F]\s+is\s+correct:\s*', re.IGNORECASE)
_BRACKETED_REFERENCE_RE     = re.compile(r'Reference:\s*(.*)', re.IGNORECASE | re.DOTALL)
_BRACKETED_BULLET_RE        = re.compile(
    r'[●•]\s*[A-F][.)]\s*(.*?)\s*[—–-]\s*(.*?)(?=[●•]|\Z)', re.DOTALL
)


def _parse_answers_bracketed(text):
    """Raw per-question data for the bracketed answer key — deliberately does
    NOT keep this format's own option letters (see the format note above);
    only the wording is carried forward for _resolve_bracketed_answer to match
    against the question paper's own lettering.
    """
    raw = {}
    headers = list(_BRACKETED_ANSWER_HEADER_RE.finditer(text))

    for index, header in enumerate(headers):
        block_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[header.end():block_end]

        why_match = _BRACKETED_WHY_RE.search(block)
        if not why_match:
            continue

        ref_match = _BRACKETED_REFERENCE_RE.search(block[:why_match.start()])
        reference = clean_text(ref_match.group(1)) if ref_match else None

        bullet_mark  = re.search(r'[●•]', block[why_match.end():])
        why_end      = why_match.end() + bullet_mark.start() if bullet_mark else len(block)
        why_text     = clean_text(block[why_match.end():why_end])

        bullets = [
            (clean_text(label), clean_text(explanation))
            for label, explanation in _BRACKETED_BULLET_RE.findall(block[why_end:])
        ]
        if bullets:
            labels, explanations = zip(*bullets)
            bullets = list(zip(labels, _trim_runaway_tail(list(explanations))))

        raw[int(header.group(1))] = {
            "reference": reference,
            "why_text":  why_text,
            "bullets":   bullets,
        }

    return raw


# A running section/batch title repeated in these source exports (e.g. "AMC
# CAT 1 Style Questions — Batch 1 Answers (Q21–Q50)"), always its own literal
# anchor and always the true start of the leaked run — unlike the vocabulary
# below, this needs no junction-walking to locate where the leak begins.
_RUNNING_HEADER_RE = re.compile(r'\bAMC\s+CAT[\s-]*\d*\b', re.IGNORECASE)

_DRAFTING_LEAK_RE = re.compile(
    r'\b(?:batch|numbering|answer key|regenerate|shuffled letters|'
    r'consecutive same-letter|topic/diagnosis repeat|recycled stock phrase)\b',
    re.IGNORECASE,
)


def _trim_runaway_tail(parts):
    """Guard against these LLM-authored papers occasionally leaking a stray
    paragraph of drafting chatter, or a running batch-title header repeated
    between sections, onto the very last option/explanation of a block
    (observed at export batch boundaries — e.g. "Continuing the numbering as
    Q51-Q100..." or "AMC CAT 1 Style Questions — Batch 1 Answers (Q21–Q50)"
    glued straight onto the prior question's last option, no punctuation in
    between). The last item in a block runs to the next header with nothing
    bounding its far end, so any such leak lands entirely on it; every other
    item is bounded on both sides and unaffected.

    Vocabulary like "batch" or "answer key" never belongs in an actual
    clinical rationale, so wherever it turns up, the leak is walked back to
    the last point a lowercase word runs straight into a capitalised one with
    no sentence break — that join is where the drafting aside started. This
    heuristic can overshoot when the leaked text is itself a multi-word title
    ("AMC CAT 1 Style Questions" has its own internal lowercase-to-capital
    joins), so the running-header anchor is checked first and, when found,
    cuts right at its own start rather than walking backward through it.
    Separately, and regardless of wording, an item wildly longer than its
    siblings gets capped, since a genuine option is never that much longer
    than the others in its own block.
    """
    if len(parts) < 2:
        return parts
    parts = list(parts)
    last  = parts[-1]

    header_leak = _RUNNING_HEADER_RE.search(last)
    if header_leak and header_leak.start() > 15:
        last = last[:header_leak.start()].rstrip()
    else:
        leak = _DRAFTING_LEAK_RE.search(last)
        if leak and leak.start() > 15:
            cut = 0
            for junction in re.finditer(r'[a-z](\s)(?=[A-Z])', last[:leak.start()]):
                cut = junction.start(1)
            if cut:
                last = last[:cut].rstrip()

    other_lengths = [len(p) for p in parts[:-1]]
    avg   = sum(other_lengths) / len(other_lengths)
    limit = max(avg * 2.0, avg + 80, 120)
    if len(last) > limit:
        space_cut = last.rfind(' ', 0, int(limit))
        last = last[:space_cut if space_cut > 0 else int(limit)].rstrip(' -—')

    parts[-1] = last
    return parts


_MATCH_STOPWORDS = frozenset("""
    a an the of is are was were be been being to and or but if then so than as at by for
    from in into on onto with without within over under above below between during before
    after this that these those it its his her their there here what which who whom when
    where why how not no nor do does did doing has have had having will would shall should
    can could may might must most more less very also each other same only just per one
    two three
""".split())


def _tokenize_for_matching(text):
    """Lowercased words, stopwords and crude plurals stripped, for comparing an
    option's wording against a chunk of the answer key's prose.

    Single-character tokens are kept — they're the only thing that
    distinguishes options like "Hepatitis A/B/C infection", so dropping them
    as noise erases the one signal that matters for that question.
    """
    tokens = []
    for word in re.findall(r'[a-z0-9]+', (text or "").lower()):
        if word in _MATCH_STOPWORDS:
            continue
        if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
            word = word[:-1]
        tokens.append(word)
    return tokens


def _weighted_overlap(option_tokens, weights, target_positions):
    """Fraction of option_tokens' weight found in target_positions (a token ->
    last-occurrence-index map), plus the latest position among only the
    DISTINCTIVE matches (weight >= 0.5) — used to break ties between options
    that share most of their wording (e.g. two dose options both ending
    "mg/24 hours") in favour of whichever distinct value the target text
    mentions last, since these explanations state background numbers before
    concluding with the actual answer.
    """
    if not option_tokens:
        return 0.0, -1
    total = sum(weights[t] for t in option_tokens)
    if total == 0:
        return 0.0, -1
    matched, last_distinctive_pos = 0.0, -1
    for token in option_tokens:
        pos = target_positions.get(token)
        if pos is None:
            continue
        matched += weights[token]
        if weights[token] >= 0.5:
            last_distinctive_pos = max(last_distinctive_pos, pos)
    return matched / total, last_distinctive_pos


def _resolve_bracketed_answer(options, entry):
    """Pair one question's options (as lettered in the question paper) against
    its answer-key entry (as worded in the answer paper), matching by wording
    rather than by letter — see the bracketed-format note above for why the
    letters can't be trusted.

    Each option's words are weighted by 1 / (how many of this question's OTHER
    options also use that word), so template phrasing shared by every choice
    (e.g. every option here ending "mg/24 hours") barely counts, and the words
    that actually distinguish one option from its siblings decide the match.

    The "why is correct" paragraph alone isn't a safe signal: it often mentions
    a wrong option's wording too, to contrast against it (e.g. "...distinct
    from dementia's chronic decline" when ruling OUT a dementia option). So
    each option's score is *why-paragraph match minus its best match against
    any "wrong" bullet* — an option that's actually correct is rarely also
    the closest match to one of the bullets explaining why some other choice
    is wrong. Ties (rounded, to dodge float noise) fall back to whichever
    option's distinctive wording appears latest in the why-paragraph, since
    these explanations tend to state background before concluding with the
    actual answer (e.g. "his oral dose is 60mg, so the SC dose is ~30mg").
    """
    letters    = list(options)
    opt_tokens = {letter: _tokenize_for_matching(options[letter]) for letter in letters}

    doc_freq = Counter()
    for letter in letters:
        for token in set(opt_tokens[letter]):
            doc_freq[token] += 1
    weights = {token: 1.0 / count for token, count in doc_freq.items()}

    why_tokens    = _tokenize_for_matching(entry.get("why_text"))
    why_positions = {token: i for i, token in enumerate(why_tokens)}
    why_scored    = {
        letter: _weighted_overlap(opt_tokens[letter], weights, why_positions)
        for letter in letters
    }

    bullets = list(entry.get("bullets") or [])
    bullet_positions = []
    for label, explanation in bullets:
        tokens = _tokenize_for_matching(f"{label} {explanation}")
        bullet_positions.append({t: i for i, t in enumerate(tokens)})

    best_bullet_score = {}
    for letter in letters:
        best = 0.0
        for positions in bullet_positions:
            score, _ = _weighted_overlap(opt_tokens[letter], weights, positions)
            best = max(best, score)
        best_bullet_score[letter] = best

    net_scored = {
        letter: (round(why_scored[letter][0] - best_bullet_score[letter], 6), why_scored[letter][1])
        for letter in letters
    }
    correct_letter = max(net_scored, key=net_scored.get)

    # Whichever remaining options best match a "wrong" bullet get that
    # bullet's explanation; matched greedily, most confident pair first, so a
    # weak/ambiguous bullet doesn't steal an option a stronger bullet wants.
    remaining_letters = [letter for letter in letters if letter != correct_letter]
    candidates = []
    for bullet_index, positions in enumerate(bullet_positions):
        for letter in remaining_letters:
            score, _ = _weighted_overlap(opt_tokens[letter], weights, positions)
            candidates.append((score, bullet_index, letter))
    candidates.sort(key=lambda c: c[0], reverse=True)

    option_explanations = {}
    used_bullets, used_letters = set(), set()
    for score, bullet_index, letter in candidates:
        if bullet_index in used_bullets or letter in used_letters:
            continue
        option_explanations[letter] = bullets[bullet_index][1]
        used_bullets.add(bullet_index)
        used_letters.add(letter)

    explanation = entry.get("why_text") or None
    if entry.get("reference"):
        explanation = f"Reference: {entry['reference']}\n\n{explanation}" if explanation else f"Reference: {entry['reference']}"

    return {
        "correct_answer":      correct_letter,
        "explanation":         explanation,
        "option_explanations": option_explanations or None,
    }


def _resolve_bracketed_answers(raw_answers, questions_by_number):
    answers = {}
    for number, entry in raw_answers.items():
        question = questions_by_number.get(number)
        if not question or not question.get("options"):
            continue
        answers[number] = [_resolve_bracketed_answer(question["options"], entry)]
    return answers


# ── Reviewed format ──────────────────────────────────────────────────────────
# A fourth answer-key layout, paired with the bracketed question paper above:
#
#   Q1. Answer: B — Autism spectrum disorder
#   A. Incorrect — ODD involves hostile, defiant behaviour...
#   B. Correct — social difficulty making friends, restricted/repetitive
#   play... (Therapeutic Guidelines Australia — Autism Spectrum Disorder)
#   C. Incorrect — ADHD is characterised by inattention...
#   ...
#
# Unlike the bracketed answer key, every option is relisted under the SAME
# letter as the question paper (confirmed by cross-checking wording), so this
# merges by letter directly — no wording match needed.

# Not anchored to the start of a line: earlier batches print this whole answer
# key as continuous wrapped prose with no paragraph break between questions,
# so a header or option marker lands at the start of a text line only by the
# accident of where the previous sentence happened to wrap. Anchoring to `^`
# silently swallowed every question whose header didn't land on a fresh line
# into its predecessor's block.
#
# Not even anchored to a preceding whitespace character: some source exports
# glue this header directly onto the closing ")" of the previous question's
# last reference citation with no space or newline at all ("...Recognition)
# Q9. [Cardiology]..."). The "Q<n>. ... Answer: <letter>" shape (optionally
# with the bracketed category/difficulty tag in between, echoing the
# bracketed question paper's own header) is distinctive enough to match
# anywhere without a false hit.
#
# The header itself also has two variants: "Q1. Answer: B" all on one line, or
# a "Q1. [Category] — Hard" line followed by "Answer: B — ..." on the next
# line — and that second line sometimes carries a wrapped tag ("— Hard —
# General practice setting —\nImage-based") or even a page-footer line
# ("Page 5Dr Solosailor's...") glued in before "Answer:" ever appears. Rather
# than modelling every shape of that in-between clutter, this skips up to 160
# characters of *anything* looking for the next "Answer: <letter>" — short
# enough that it can't leap over a real gap into a different question's
# answer line (the shortest real per-question block is far longer than that),
# but long enough to clear the longest observed clutter.
_REVIEWED_ANSWER_HEADER_RE = re.compile(r'Q(\d+)\.[\s\S]{0,160}?Answer:\s*([A-F])\b')
# The separator after "Correct"/"Incorrect" is a dash in some papers and a
# plain period in others ("A. Incorrect. X-ray adds..." vs "A. Incorrect —
# X-ray adds...") — consuming only an optional dash left that period sitting
# on the front of every explanation. `[.\s—–-]*` eats any run of those
# separator characters regardless of which one the source used.
_REVIEWED_OPTION_MARKER_RE = re.compile(r'(?:(?<=\s)|^)([A-F])\.\s*(Correct|Incorrect)\b[.\s—–-]*')


def _parse_answers_reviewed(text):
    """One option is occasionally left out of the list entirely (seen from a
    certain question on in one paper) — that option just ends up with no
    wrong-answer explanation. The correct answer itself is unaffected, since
    it's read from the header rather than inferred from which options got a
    bullet.
    """
    answers = {}
    headers = list(_REVIEWED_ANSWER_HEADER_RE.finditer(text))

    for index, header in enumerate(headers):
        block_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[header.end():block_end]
        header_letter = header.group(2)

        markers = list(_REVIEWED_OPTION_MARKER_RE.finditer(block))
        if not markers:
            continue

        entries = []
        for i, marker in enumerate(markers):
            seg_end = markers[i + 1].start() if i + 1 < len(markers) else len(block)
            entries.append({
                "letter":      marker.group(1),
                "status":      marker.group(2),
                "explanation": clean_text(block[marker.end():seg_end]),
            })

        # Every question's last entry runs to the next header with nothing
        # else bounding its far end, so a running header/footer sitting
        # between batches with no blank line before it ("AMC CAT 1 Style
        # Questions — Batch 1 Answers (Q51–80)") lands glued onto whichever
        # question happens to be last before it — see _trim_runaway_tail.
        if len(entries) >= 2:
            entries[-1]["explanation"] = _trim_runaway_tail(
                [e["explanation"] for e in entries]
            )[-1]

        correct_entries = [e for e in entries if e["status"] == "Correct"]
        if len(correct_entries) == 1:
            correct_entry = correct_entries[0]
        elif len(correct_entries) > 1:
            # The source occasionally revises its own answer mid-explanation
            # ("B. Correct (revised) — ..." after an earlier "A. Correct —"
            # it then contradicts). Prefer whichever entry says so, else
            # whichever matches the header, else the last one — a later
            # entry corrects an earlier one here, never the reverse.
            correct_entry = (
                next((e for e in correct_entries if 'revised' in e["explanation"].lower()), None)
                or next((e for e in correct_entries if e["letter"] == header_letter), None)
                or correct_entries[-1]
            )
        else:
            correct_entry = next((e for e in entries if e["letter"] == header_letter), None)

        if not correct_entry:
            continue

        option_explanations = {
            e["letter"]: e["explanation"]
            for e in entries
            if e is not correct_entry and e["explanation"]
        }

        answers[int(header.group(1))] = [{
            "correct_answer":      correct_entry["letter"],
            "explanation":         correct_entry["explanation"] or None,
            "option_explanations": option_explanations or None,
        }]

    return answers


# ── Concise format ───────────────────────────────────────────────────────────
# A fifth answer-key layout:
#
#   Q1. Correct: A A) Correct — Start with a low replacement dose (25–50 mcg)
#   and titrate based on repeat TSH... B) Ultrasound is not required to
#   diagnose primary hypothyroidism... C) ... D) ... E) ... Reference: RACGP –
#   Thyroid disease: diagnosis and management in general practice.
#
# The correct letter is only ever given once, in the header ("Correct: A") —
# unlike the reviewed format, wrong options carry no "Incorrect" keyword at
# all, they just state the critique directly, so the correct option is found
# by matching the header's letter rather than by scanning option text for a
# status word. Options are lettered the same way as the paired bracketed
# question paper (confirmed by wording), so this merges by letter directly.

_LETTERED_ANSWER_HEADER_RE = re.compile(r'Q(\d+)\.\s*Correct:\s*([A-F])\b')
_LETTERED_REFERENCE_RE     = re.compile(r'\bReference:\s*(.*)', re.IGNORECASE | re.DOTALL)


def _parse_answers_lettered(text):
    answers = {}
    headers = list(_LETTERED_ANSWER_HEADER_RE.finditer(text))

    for index, header in enumerate(headers):
        block_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = ' '.join(text[header.end():block_end].split())
        correct_letter = header.group(2)

        # The reference note trails the last option with nothing else
        # bounding it, so it's sliced off before splitting options — left in
        # place it would otherwise read as the tail of the last option's text.
        ref_match = _LETTERED_REFERENCE_RE.search(block)
        reference = clean_text(ref_match.group(1)) if ref_match else None
        opts_block = block[:ref_match.start()] if ref_match else block

        _, options = _split_inline_options(opts_block)
        if len(options) < 2 or correct_letter not in options:
            continue

        correct_text = re.sub(
            r'^Correct\s*[—:-]*\s*', '', options[correct_letter], flags=re.IGNORECASE
        ).strip()
        option_explanations = {
            letter: opt_text for letter, opt_text in options.items() if letter != correct_letter
        }

        explanation = correct_text or None
        if reference:
            explanation = f"Reference: {reference}\n\n{explanation}" if explanation else f"Reference: {reference}"

        answers[int(header.group(1))] = [{
            "correct_answer":      correct_letter,
            "explanation":         explanation,
            "option_explanations": option_explanations or None,
        }]

    return answers


# ── Format dispatch ──────────────────────────────────────────────────────────

def parse_questions(text, page_starts=None):
    """Parse any of the three paper layouts.

    Each is tried in turn and only falls back when it yields nothing, so a
    paper that parsed before keeps parsing exactly as it did.
    """
    labelled = _parse_questions_labelled(text)
    if labelled:
        print(f"[parser] questions: labelled format, {len(labelled)} found")
        return labelled

    bracketed = _parse_questions_bracketed(text, page_starts)
    if bracketed:
        print(f"[parser] questions: bracketed format, {len(bracketed)} found")
        return bracketed

    prose = _parse_questions_prose(text, page_starts)
    print(f"[parser] questions: prose format, {len(prose)} found")
    return prose


def parse_answers(text, questions_by_number=None):
    """Parse any of the four answer-key layouts.

    The bracketed format needs the parsed questions (to match option wording
    against) — `questions_by_number` is only required for that path.
    """
    labelled = _parse_answers_labelled(text)
    if labelled:
        print(f"[parser] answers: labelled format, {len(labelled)} found")
        return labelled

    reviewed = _parse_answers_reviewed(text)
    if reviewed:
        print(f"[parser] answers: reviewed format, {len(reviewed)} found")
        return reviewed

    lettered = _parse_answers_lettered(text)
    if lettered:
        print(f"[parser] answers: lettered format, {len(lettered)} found")
        return lettered

    bracketed_raw = _parse_answers_bracketed(text)
    if bracketed_raw:
        resolved = _resolve_bracketed_answers(bracketed_raw, questions_by_number or {})
        print(f"[parser] answers: bracketed format, {len(resolved)} found")
        return resolved

    prose = _parse_answers_prose(text)
    print(f"[parser] answers: prose format, {len(prose)} found")
    return prose


def merge_questions_answers(questions, answers, images_by_page=None,
                            answer_page_map=None, answers_images=None):
    """
    Cloudinary URLs are already absolute (https://res.cloudinary.com/...) so
    base_url is no longer needed.

    `answers` maps question_number -> list of answer entries in document
    order. Some source papers bundle more than one exam session into a single
    PDF pair with overlapping numbering (the same "Q166" printed once per
    session, each time with unrelated content), so a plain "look up by
    number" would silently pair a question with the wrong session's answer.
    Popping each number's answers in the order they were found instead pairs
    the Nth occurrence of a number in the questions with the Nth occurrence
    in the answers — correct as long as both PDFs list their sessions in the
    same relative order, which is how a paired question/answer set is put
    together. A question number with no (or no more) queued answers is left
    unanswered rather than guessed at.
    """
    merged = []
    answer_queues = {n: deque(entries) for n, entries in answers.items()}

    for q in questions:
        q_no        = q["question_number"]
        queue       = answer_queues.get(q_no)
        answer_data = queue.popleft() if queue else {}

        question_images = None
        if images_by_page and q.get("page_number"):
            paths = images_by_page.get(q["page_number"])
            question_images = paths if paths else None   # already full URLs

        answer_images = None
        if answers_images and answer_page_map:
            ans_page = answer_page_map.get(q_no)
            if ans_page:
                paths = answers_images.get(ans_page)
                answer_images = paths if paths else None  # already full URLs

        correct_answer          = answer_data.get("correct_answer")
        raw_option_explanations = answer_data.get("option_explanations") or {}
        options                 = q.get("options", {})

        formatted_options = [
            {
                "option_key":  letter,
                "option_text": text,
                "is_correct":  letter == correct_answer,
                "explanation": strip_option_prefix(raw_option_explanations.get(letter), text),
            }
            for letter, text in options.items()
        ]

        # Some source papers skip a letter (e.g. a question's options run
        # A, B, D, E with no C printed anywhere). is_correct and explanation
        # above are already resolved per option object, not looked up again
        # by letter, so relabelling option_key to run A, B, C, D... here is
        # purely cosmetic — it cannot disturb which option is marked correct
        # or which explanation belongs to which. A no-op when the source
        # letters were already consecutive.
        for position, option in enumerate(formatted_options):
            if position < len(OPTION_LETTERS):
                option["option_key"] = OPTION_LETTERS[position]
        correct_answer = next((o["option_key"] for o in formatted_options if o["is_correct"]), None)

        merged.append({
            **q,
            "options":       formatted_options,
            "images":        question_images,
            "correct_answer":correct_answer,
            "explanation":   answer_data.get("explanation"),
            "answer_images": answer_images,
        })

    return merged


# ==========================================
# BACKEND HELPERS
# ==========================================
async def create_batch(client: httpx.AsyncClient, title: str) -> int:
    resp = await client.post(
        f"{BACKEND_URL}/api/admin/import-batches",
        json={"title": title},
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["batch_id"]


async def send_questions(client: httpx.AsyncClient, batch_id: int, questions: list):
    resp = await client.post(
        f"{BACKEND_URL}/api/admin/import-batches/{batch_id}/receive",
        json={
            "status":          "success",
            "total_questions": len(questions),
            "questions":       questions,
        },
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ==========================================
# IMPORT ENDPOINT
# ==========================================
@app.post("/import")
async def import_pdfs(
    questions_pdf: UploadFile = File(...),
    answers_pdf:   UploadFile = File(...),
    title:    Optional[str] = Form(None),
    batch_id: Optional[int] = Form(None),
):
    if not batch_id and not title:
        raise HTTPException(status_code=400, detail="Provide either batch_id or title")

    if not questions_pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="questions_pdf must be a PDF")

    if not answers_pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="answers_pdf must be a PDF")

    questions_path, answers_path = None, None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as q_temp:
            q_temp.write(await questions_pdf.read())
            questions_path = q_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as a_temp:
            a_temp.write(await answers_pdf.read())
            answers_path = a_temp.name

        job_id = str(uuid.uuid4())

        questions_text, questions_pages = extract_pdf_text_pages(questions_path)
        answers_text      = extract_pdf_text(answers_path)
        questions_images  = extract_pdf_images(questions_path, job_id, prefix="q")
        answers_images    = extract_pdf_images(answers_path,   job_id, prefix="a")
        answer_page_map   = get_answer_page_map(answers_path)

        parsed_questions = parse_questions(questions_text, questions_pages)
        questions_by_number = {q["question_number"]: q for q in parsed_questions}
        parsed_answers   = parse_answers(answers_text, questions_by_number)

        final_questions = merge_questions_answers(
            parsed_questions,
            parsed_answers,
            questions_images,
            answer_page_map,
            answers_images,
        )

        async with httpx.AsyncClient() as client:
            if not batch_id:
                batch_id = await create_batch(client, title)
            await send_questions(client, batch_id, final_questions)

        # A question with no entry in the answer paper imports with no correct
        # option, so every student answering it is marked wrong. Reported rather
        # than dropped: the question is still real, and the gap is usually a
        # missing page in the answers PDF worth fixing at the source.
        unanswered = [
            q["question_number"]
            for q in final_questions
            if not any(o.get("is_correct") for o in q.get("options", []))
        ]

        return {
            "status":                "success",
            "batch_id":              batch_id,
            "total_questions":       len(final_questions),
            "questions_without_answer": unanswered,
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {e.response.text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if questions_path and os.path.exists(questions_path):
            os.remove(questions_path)
        if answers_path and os.path.exists(answers_path):
            os.remove(answers_path)


# ==========================================
# QBANK IMPORT (single screenshot PDF → Claude vision)
# ==========================================
#
# The recall /import above parses a text PDF pair with regex. QBank source PDFs
# are a different animal: exports of another quiz app, one screenshot per page,
# no selectable text at all. Each question spans two pages —
#
#   odd page  : the stem + options with empty radio circles
#   even page : the same stem + every option's "CORRECT."/"INCORRECT."
#               explanation, a check icon on the right answer
#
# so the even pages carry everything. Each even page is sent to Claude, which
# returns the stem, the options in order, which one is correct, each
# explanation, and a best-guess subject. Results are merged by question number
# and pushed to the same /receive endpoint the recall importer uses.

OPTION_KEYS = "ABCDEFGH"

# Kept short and closed so findOrCreateSubject on the backend does not accumulate
# a dozen spellings of "O&G". The admin can still retag anything in the batch
# editor.
QBANK_SUBJECTS = [
    "Medicine",
    "Surgery",
    "Obstetrics & Gynaecology",
    "Paediatrics",
    "Psychiatry",
    "General Practice",
    "Emergency Medicine",
    "Ethics & Law",
    "Population Health",
]

# One closed topic list per subject, for the same reason QBANK_SUBJECTS is
# closed: thousands of questions tagged free-form end up under a dozen
# spellings of each topic, and students filter by topic. Edit here; the admin
# can still retag a question in the batch editor.
QBANK_TOPICS = {
    "Medicine": [
        "Cardiology", "Respiratory", "Gastroenterology & Hepatology",
        "Endocrinology & Diabetes", "Nephrology", "Neurology", "Haematology",
        "Oncology", "Infectious Diseases", "Rheumatology & Immunology",
        "Dermatology", "Geriatric Medicine", "Palliative Care",
        "Clinical Pharmacology",
    ],
    "Surgery": [
        "General Surgery", "Upper GI & Hepatobiliary", "Colorectal",
        "Breast & Endocrine Surgery", "Vascular Surgery", "Urology",
        "Orthopaedics", "Neurosurgery", "Trauma", "ENT", "Ophthalmology",
        "Plastic Surgery & Burns", "Anaesthesia & Perioperative Care",
    ],
    "Obstetrics & Gynaecology": [
        "Antenatal Care", "Complications of Pregnancy", "Labour & Delivery",
        "Postnatal Care", "Gynaecology", "Contraception & Fertility",
        "Gynaecological Oncology", "Menopause",
    ],
    "Paediatrics": [
        "Neonatology", "Growth & Development", "Infections & Immunisation",
        "Paediatric Emergencies", "Paediatric Medicine", "Paediatric Surgery",
        "Child Protection", "Adolescent Health",
    ],
    "Psychiatry": [
        "Mood Disorders", "Anxiety Disorders", "Psychotic Disorders",
        "Substance Use", "Personality Disorders", "Child & Adolescent Psychiatry",
        "Old Age Psychiatry & Dementia", "Eating Disorders", "Psychopharmacology",
        "Risk Assessment & Mental Health Law",
    ],
    "General Practice": [
        "Preventive Health & Screening", "Chronic Disease Management",
        "Common Presentations", "Sexual Health", "Travel & Occupational Health",
        "Aboriginal & Torres Strait Islander Health",
    ],
    "Emergency Medicine": [
        "Resuscitation", "Shock & Sepsis", "Acute Cardiorespiratory",
        "Acute Neurology", "Toxicology & Envenomation",
        "Environmental Emergencies", "Emergency Procedures", "Acute Abdomen",
    ],
    "Ethics & Law": [
        "Consent & Capacity", "Confidentiality", "Professionalism",
        "End-of-Life Care", "Mandatory Reporting",
    ],
    "Population Health": [
        "Epidemiology & Biostatistics", "Screening Programs",
        "Public Health & Immunisation", "Evidence-Based Medicine",
    ],
}

_QBANK_PROMPT = (
    "You are extracting one multiple-choice medical exam question from a screenshot "
    "of an online question bank (AMC or eMedici style).\n\n"
    "A page is ANSWERED when the correct option is marked — highlighted in green "
    "with a tick, and/or each option is followed by an explanation (which may begin "
    '"CORRECT."/"INCORRECT." or may be a discussion paragraph). Otherwise it is '
    "UNANSWERED (plain radio circles, no explanations). A page may also be a cover "
    "or blank.\n\n"
    "Reply with ONLY a JSON object, no markdown fences, exactly this shape:\n"
    "{\n"
    '  "has_answer": <true only if the correct option is marked or explanations are shown>,\n'
    '  "question_number": <integer after the word "Question", or null>,\n'
    '  "question_text": "<the full case/scenario, INCLUDING any investigation '
    'results and the final lead-in question sentence, as one string>",\n'
    f'  "subject": <one of {QBANK_SUBJECTS} or null>,\n'
    '  "topic": <one topic from the list for the chosen subject (below), or null '
    'if none fits>,\n'
    '  "options": [\n'
    '    { "text": "<option text exactly as shown, no leading letter>",\n'
    '      "is_correct": <true only for the green / ticked / "CORRECT" option>,\n'
    '      "explanation": "<why this option is right or wrong: the sentence(s) or '
    'paragraph about THIS option, with any leading CORRECT./INCORRECT. word '
    'removed; null if none shown>" }\n'
    "  ],\n"
    '  "key_points": [<short take-home learning points shown on the page (often '
    'highlighted / bulleted near the bottom); [] if none>],\n'
    '  "has_image": <true if a clinical figure is embedded in the page — an X-ray, '
    "CT/MRI/ultrasound, ECG, clinical photograph, pathology slide, chart or "
    "diagram. NOT buttons, icons, avatars or the answer-percentage bars>,\n"
    '  "image_region": <[left, top, right, bottom] as fractions 0.0-1.0 of the '
    "whole page. Box ONLY the figure itself — stop at its bottom edge, before the "
    '"Choose the single best answer" line, the question text or any options. null '
    "when has_image is false>\n"
    "}\n\n"
    "Rules:\n"
    "- Eye, ENT and orthopaedic questions are Surgery, whatever the setting.\n"
    "- Topics by subject: "
    + "; ".join(f"{subject}: {', '.join(topics)}" for subject, topics in QBANK_TOPICS.items())
    + ".\n"
    "- List options top to bottom in the order shown.\n"
    "- On an answered page exactly one option has is_correct true.\n"
    "- If has_answer is false, still list the option texts (is_correct false, "
    "explanation null).\n"
    "- Never invent an explanation or a correct answer that is not shown. If the "
    "page is not a question at all, return has_answer false, question_number null, "
    "empty options, has_image false."
)


def _render_page_png(pdf_path: str, page_number: int, dpi: int) -> bytes:
    """One page of the PDF as PNG bytes, at the given DPI."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        return pix.tobytes("png")
    finally:
        doc.close()


def _render_page_region_png(pdf_path: str, page_number: int, frac_rect: tuple,
                            dpi: int = 300, trim: bool = True) -> bytes:
    """A rectangular region of a page (given as 0-1 fractions) as PNG bytes.

    Higher DPI for the final crop because these are clinical figures a student
    will zoom into. `trim` removes near-white page margins from the result.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        r = page.rect
        x0, y0, x1, y1 = frac_rect
        clip = fitz.Rect(
            r.x0 + x0 * r.width, r.y0 + y0 * r.height,
            r.x0 + x1 * r.width, r.y0 + y1 * r.height,
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=clip)
        png = pix.tobytes("png")
        return _trim_whitespace(png) if trim else png
    finally:
        doc.close()


def _figure_bbox(png_bytes: bytes):
    """Locate the clinical figure on a rendered page by its pixel texture.

    A page of text — including the dense explanation block on an eMedici answer
    page — is near-white with thin dark strokes: very few mid-grey pixels per
    row. An X-ray, scan or clinical photo is saturated with mid-grey. Measured
    across real pages the figure band sits around a 0.40 mid-tone fraction and
    every text band well under 0.20, so a 0.32 cut cleanly separates them (and
    returns nothing on a page that has no figure).

    Returns (left, top, right, bottom) as fractions of the page, or None.
    """
    try:
        from io import BytesIO
        from PIL import Image

        g = Image.open(BytesIO(png_bytes)).convert("L")
        small = g.resize((160, 720))
        sw, sh = small.size
        px = small.load()

        LO, HI = 30, 222

        def row_mid(y):
            return sum(1 for x in range(sw) if LO < px[x, y] < HI) / sw

        rows = [row_mid(y) for y in range(sh)]

        # The figure's signature is a long run of rows that ALL carry heavy
        # mid-tone — an X-ray/photo has no internal white gaps, whereas text
        # (however dense) has a near-white row between every line. Allow only a
        # lone gap row (a thin caption or label inside the image) to be bridged.
        solid = [1 if r >= 0.28 else 0 for r in rows]
        for y in range(1, sh - 1):
            if not solid[y] and solid[y - 1] and solid[y + 1]:
                solid[y] = 1

        best = (0, 0)
        cur = None
        for y, v in enumerate(solid + [0]):
            if v and cur is None:
                cur = y
            elif not v and cur is not None:
                if (y - cur) > (best[1] - best[0]):
                    best = (cur, y)
                cur = None
        top, bot = best

        if (bot - top) < 0.08 * sh:
            return None
        if (sum(rows[top:bot]) / (bot - top)) < 0.33:
            return None

        # Reject a flat colour fill (e.g. a solid footer band): a real figure
        # has plenty of tonal variation.
        from PIL import ImageStat
        band = small.crop((0, top, sw, bot))
        if ImageStat.Stat(band).stddev[0] < 14:
            return None

        def col_mid(x):
            return sum(1 for y in range(top, bot) if LO < px[x, y] < HI) / (bot - top)

        cols = [x for x in range(sw) if col_mid(x) >= 0.18]
        left, right = (cols[0], cols[-1] + 1) if cols else (0, sw)

        m = 0.006
        return (
            max(0.0, left / sw - m), max(0.0, top / sh - m),
            min(1.0, right / sw + m), min(1.0, bot / sh + m),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[qbank] figure-bbox detection skipped: {e}")
        return None


def _trim_whitespace(png_bytes: bytes) -> bytes:
    """Crop away rows/columns that are almost entirely page-white.

    Claude's bounding box usually overshoots downward onto the question text.
    The figure itself carries dark or coloured pixels (an X-ray's black
    surround, a photo, a chart), so trimming near-white margins reliably tightens
    it. Falls back to the untrimmed image if the detection looks unsafe.
    """
    try:
        from io import BytesIO
        from PIL import Image

        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        W, H = img.size
        small = img.convert("L").resize((min(W, 240), min(H, 320)))
        sw, sh = small.size
        px = small.load()

        WHITE = 238
        ROW_WHITE_FRAC = 0.985

        def row_is_content(y):
            non_white = sum(1 for x in range(sw) if px[x, y] < WHITE)
            return non_white > (1 - ROW_WHITE_FRAC) * sw

        def col_is_content(x):
            non_white = sum(1 for y in range(sh) if px[x, y] < WHITE)
            return non_white > (1 - ROW_WHITE_FRAC) * sh

        rows = [y for y in range(sh) if row_is_content(y)]
        cols = [x for x in range(sw) if col_is_content(x)]
        if not rows or not cols:
            return png_bytes

        margin = 0.01
        top = max(0.0, rows[0] / sh - margin)
        bot = min(1.0, (rows[-1] + 1) / sh + margin)
        left = max(0.0, cols[0] / sw - margin)
        right = min(1.0, (cols[-1] + 1) / sw + margin)

        # Bail if the trim is trivial or implausible.
        if (bot - top) < 0.1 or (right - left) < 0.1:
            return png_bytes
        if (bot - top) > 0.98 and (right - left) > 0.98:
            return png_bytes

        box = (int(left * W), int(top * H), int(right * W), int(bot * H))
        out = BytesIO()
        img.crop(box).save(out, format="PNG")
        return out.getvalue()
    except Exception as e:  # noqa: BLE001
        print(f"[qbank] whitespace trim skipped: {e}")
        return png_bytes


def _pages_to_read(total_pages: int, mode: str) -> list:
    """Which page numbers to send to Claude.

    'answered' (default) sends only the even pages — the ones carrying the
    explanations in this export format. 'all' sends every page and is the
    fallback for a PDF whose parity is off (a long answer that spilled onto a
    third page shifts everything after it).
    """
    if mode == "all":
        return list(range(1, total_pages + 1))
    return list(range(2, total_pages + 1, 2))


async def _extract_page(client, sem, pdf_path: str, page_number: int) -> Optional[dict]:
    """Ask Claude to read one page. Returns the parsed dict, or None on failure."""
    async with sem:
        # Render inside the semaphore so only a few page images are held in
        # memory at once, not the whole PDF's worth.
        png = await asyncio.to_thread(_render_page_png, pdf_path, page_number, QBANK_DPI)
        b64 = base64.standard_b64encode(png).decode("ascii")

        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model=QBANK_MODEL,
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/png", "data": b64,
                            }},
                            {"type": "text", "text": _QBANK_PROMPT},
                        ],
                    }],
                )
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                # Strip a stray ```json fence if the model adds one.
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
                data = json.loads(text)
                data["_page"] = page_number
                return data
            except json.JSONDecodeError:
                if attempt == 2:
                    print(f"[qbank] page {page_number}: unparseable response, skipped")
                    return None
            except Exception as e:  # noqa: BLE001 — one bad page must not sink the run
                if attempt == 2:
                    print(f"[qbank] page {page_number}: {e}")
                    return None
                await asyncio.sleep(2 * (attempt + 1))
    return None


def _score(entry: dict) -> tuple:
    """How good an extraction is — an answered page with more explanations wins."""
    opts = entry.get("options") or []
    return (
        1 if entry.get("has_answer") else 0,
        sum(1 for o in opts if o.get("explanation")),
        len(opts),
    )


def _to_backend_question(number: int, entry: dict):
    """Map one merged extraction to the /receive question shape, or None to skip."""
    opts = entry.get("options") or []
    stem = (entry.get("question_text") or "").strip()

    if not entry.get("has_answer") or len(opts) < 2 or not stem:
        return None
    if not any(o.get("is_correct") for o in opts):
        return None

    subject = entry.get("subject")
    if subject not in QBANK_SUBJECTS:
        subject = None
    topic = entry.get("topic")
    if topic not in QBANK_TOPICS.get(subject, []):
        topic = None

    # Take-home points → the question-level explanation, shown as the summary note.
    points = [p.strip() for p in (entry.get("key_points") or []) if isinstance(p, str) and p.strip()]
    explanation = "\n".join(f"• {p}" for p in points) or None

    images = [u for u in (entry.get("_image_urls") or []) if u]

    return {
        "question_number": number,
        "question_text": stem,
        "subject": subject,
        "topic": topic,
        "source_type": "qbank",
        "question_type": "image_based" if images else "single_choice",
        "marks": 1,
        "negative_marks": 0,
        "explanation": explanation,
        **({"images": images} if images else {}),
        "options": [
            {
                "option_key": OPTION_KEYS[i] if i < len(OPTION_KEYS) else str(i + 1),
                "option_text": (o.get("text") or "").strip(),
                "is_correct": bool(o.get("is_correct")),
                "explanation": (o.get("explanation") or None),
            }
            for i, o in enumerate(opts)
        ],
    }


def _clamp_region(region, pad):
    try:
        x0, y0, x1, y1 = (float(v) for v in region)
    except (TypeError, ValueError):
        return None
    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(1.0, x1 + pad), min(1.0, y1 + pad)
    if x1 - x0 < 0.06 or y1 - y0 < 0.03:
        return None
    return (x0, y0, x1, y1)


async def _upload_question_image(http, sem, pdf_path: str, page_number: int, region) -> Optional[str]:
    """Crop the figure out of a page and store it via the backend. Returns the
    path the question row should carry, or None on any failure (non-fatal — the
    question still imports, just without the picture).

    The crop box comes from `_figure_bbox` (pixel-texture detection on a
    full-page render), which is far more reliable than the model's own bounding
    box. The model's `image_region` is only the fallback when detection finds
    nothing.
    """
    async with sem:
        try:
            full = await asyncio.to_thread(_render_page_png, pdf_path, page_number, 150)
            box = _figure_bbox(full) or _clamp_region(region, 0.03)
            if not box:
                return None

            png = await asyncio.to_thread(
                _render_page_region_png, pdf_path, page_number, box, 300, True
            )
            resp = await http.post(
                f"{BACKEND_URL}/api/admin/import-batches/images",
                files={"image": (f"q-p{page_number}.png", png, "image/png")},
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("path")
        except Exception as e:  # noqa: BLE001
            print(f"[qbank] image for page {page_number} failed: {e}")
            return None


async def _process_qbank(pdf_path: str, batch_id: int, mode: str, explicit_pages: Optional[list] = None):
    """Background job: read the PDF with Claude and push the questions to the backend.

    `explicit_pages` (1-based) reads exactly those pages and skips the gap sweep —
    used to backfill a handful of questions the first pass missed without
    re-reading the whole PDF.
    """
    try:
        import anthropic
    except ImportError:
        await _mark_batch_failed(batch_id, ["anthropic package not installed on the import service"])
        _safe_unlink(pdf_path)
        return

    if not ANTHROPIC_API_KEY:
        await _mark_batch_failed(batch_id, ["ANTHROPIC_API_KEY is not set on the import service"])
        _safe_unlink(pdf_path)
        return

    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()

        if explicit_pages:
            pages = [p for p in explicit_pages if 1 <= p <= total_pages]
        else:
            pages = _pages_to_read(total_pages, mode)
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        sem = asyncio.Semaphore(max(1, QBANK_CONCURRENCY))

        results = await asyncio.gather(*(
            _extract_page(client, sem, pdf_path, p) for p in pages
        ))

        # An answered page whose "Question N" header scrolled off the top comes
        # back with question_number null. Borrow the number from the page just
        # before it (that page carries the same question's header) when we read
        # it in this run.
        by_page = {d["_page"]: d for d in results if d}
        for data in results:
            if data and data.get("has_answer") and not isinstance(data.get("question_number"), int):
                prev = by_page.get(data["_page"] - 1)
                if prev and isinstance(prev.get("question_number"), int):
                    data["question_number"] = prev["question_number"]

        # Merge by question number, keeping the richest extraction seen.
        by_number: dict = {}
        for data in results:
            if not data:
                continue
            n = data.get("question_number")
            if not isinstance(n, int):
                continue
            if n not in by_number or _score(data) > _score(by_number[n]):
                by_number[n] = data

        # If the "answered pages only" pass left gaps, sweep the neighbouring
        # pages we skipped and merge anything new.
        if not explicit_pages and mode != "all" and by_number:
            expected = set(range(1, max(by_number) + 1))
            missing = sorted(expected - set(by_number))
            sweep = sorted({
                p for n in missing for p in (2 * n - 1, 2 * n, 2 * n + 1)
                if 1 <= p <= total_pages and p not in pages
            })
            if sweep:
                print(f"[qbank] gap sweep over {len(sweep)} pages for {missing}")
                extra = await asyncio.gather(*(
                    _extract_page(client, sem, pdf_path, p) for p in sweep
                ))
                for data in extra:
                    if not data:
                        continue
                    n = data.get("question_number")
                    if not isinstance(n, int):
                        continue
                    if n not in by_number or _score(data) > _score(by_number[n]):
                        by_number[n] = data

        async with httpx.AsyncClient() as http:
            # Crop + store every figure first, so the built questions carry image
            # paths. Best-effort — a failed upload just leaves that question
            # picture-less.
            img_sem = asyncio.Semaphore(max(1, QBANK_CONCURRENCY))
            with_images = [
                d for d in by_number.values()
                if d.get("has_image") and isinstance(d.get("_page"), int)
            ]
            if with_images:
                print(f"[qbank] uploading {len(with_images)} question images")
                urls = await asyncio.gather(*(
                    _upload_question_image(http, img_sem, pdf_path, d["_page"], d["image_region"])
                    for d in with_images
                ))
                for d, url in zip(with_images, urls):
                    if url:
                        d["_image_urls"] = [url]

            questions, skipped = [], []
            for n in sorted(by_number):
                q = _to_backend_question(n, by_number[n])
                (questions.append(q) if q else skipped.append(n))

            if not questions:
                await _mark_batch_failed(batch_id, [
                    f"Read {total_pages} pages but found no usable answered questions.",
                    "If this PDF has a cover page, or all the answers sit in a "
                    "separate block of pages, re-upload with an explicit pages "
                    "range (e.g. pages=121-240) or mode=all.",
                    f"Skipped question numbers: {skipped}" if skipped else "",
                ])
                return

            await _send_questions_with_logs(http, batch_id, questions, skipped)

        img_count = sum(1 for q in questions if q.get("images"))
        print(f"[qbank] batch {batch_id}: {len(questions)} imported ({img_count} with images), {len(skipped)} skipped")

    except Exception as e:  # noqa: BLE001
        print(f"[qbank] batch {batch_id} failed: {e}")
        await _mark_batch_failed(batch_id, [f"QBank extraction crashed: {e}"])
    finally:
        _safe_unlink(pdf_path)


def _safe_unlink(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


async def _mark_batch_failed(batch_id: int, logs: list):
    try:
        async with httpx.AsyncClient() as http:
            await http.post(
                f"{BACKEND_URL}/api/admin/import-batches/{batch_id}/receive",
                json={"status": "failed", "total_questions": 0, "questions": [],
                      "logs": [l for l in logs if l]},
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                timeout=30,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[qbank] could not mark batch {batch_id} failed: {e}")


async def _send_questions_with_logs(http, batch_id: int, questions: list, skipped: list):
    resp = await http.post(
        f"{BACKEND_URL}/api/admin/import-batches/{batch_id}/receive",
        json={
            "status": "success",
            "total_questions": len(questions) + len(skipped),
            "questions": questions,
            "logs": ([f"Skipped question numbers (no clear answer page): {skipped}"]
                     if skipped else []),
        },
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


@app.post("/import/qbank")
async def import_qbank(
    background: BackgroundTasks,
    pdf:   UploadFile = File(...),
    title: str        = Form(...),
    mode:  str        = Form("answered"),   # "answered" | "all"
    pages: Optional[str] = Form(None),      # e.g. "41,42,43" — read exactly these
):
    """Turn a screenshot-export MCQ PDF into a qbank import batch.

    Returns as soon as the batch row exists; the pages are read from Claude in
    the background (10–20 min for ~400 pages). Watch the batch on the admin
    Import Batches page — it flips from Processing to Completed / Failed.

    Pass `pages` to read only a specific set — comma-separated, 1-based, ranges
    allowed (e.g. "41,42,43" or "121-240"). Use this when the PDF keeps all its
    answers in a separate block of pages (eMedici: questions 1–N, then answers
    N+1–2N), or to backfill a few questions the first pass missed at a few cents.
    """
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="pdf must be a .pdf file")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the import service")
    if mode not in ("answered", "all"):
        raise HTTPException(status_code=400, detail="mode must be 'answered' or 'all'")

    explicit_pages = None
    if pages:
        try:
            picked = set()
            for part in pages.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    lo, hi = (int(x) for x in part.split("-", 1))
                    picked.update(range(min(lo, hi), max(lo, hi) + 1))
                else:
                    picked.add(int(part))
            explicit_pages = sorted(picked)
        except ValueError:
            raise HTTPException(status_code=400, detail="pages must be integers or ranges like 121-240")
        if not explicit_pages:
            raise HTTPException(status_code=400, detail="pages was empty")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(await pdf.read())
    tmp.close()

    try:
        doc = fitz.open(tmp.name)
        page_count = doc.page_count
        doc.close()
    except Exception:
        _safe_unlink(tmp.name)
        raise HTTPException(status_code=400, detail="Could not open that PDF")

    try:
        async with httpx.AsyncClient() as http:
            batch_id = await create_batch(http, title)
    except httpx.HTTPStatusError as e:
        _safe_unlink(tmp.name)
        raise HTTPException(status_code=502, detail=f"Backend error creating batch: {e.response.text}")

    background.add_task(_process_qbank, tmp.name, batch_id, mode, explicit_pages)

    return {
        "status": "processing",
        "batch_id": batch_id,
        "pdf_pages": page_count,
        "reading_pages": len(explicit_pages) if explicit_pages else None,
        "message": "Extraction started. Watch the batch status in the admin panel.",
    }
