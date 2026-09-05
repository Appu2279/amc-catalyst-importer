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
from collections import deque

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
            page_text = page.extract_text()
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
            page_text = page.extract_text()
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


def get_answer_page_map(pdf_path):
    """Returns {question_number: page_number} by scanning each page of the answers PDF."""
    page_map = {}
    qn_pattern = re.compile(r'Question Number:\s*(\d+)', re.IGNORECASE)
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            for match in qn_pattern.finditer(page_text):
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
            "negative_marks":  1,
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
            "negative_marks":  1,
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


# ── Format dispatch ──────────────────────────────────────────────────────────

def parse_questions(text, page_starts=None):
    """Parse either paper layout.

    The labelled format is tried first and only falls back when it yields
    nothing, so a paper that parsed before keeps parsing exactly as it did.
    """
    labelled = _parse_questions_labelled(text)
    if labelled:
        print(f"[parser] questions: labelled format, {len(labelled)} found")
        return labelled

    prose = _parse_questions_prose(text, page_starts)
    print(f"[parser] questions: prose format, {len(prose)} found")
    return prose


def parse_answers(text):
    labelled = _parse_answers_labelled(text)
    if labelled:
        print(f"[parser] answers: labelled format, {len(labelled)} found")
        return labelled

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
        parsed_answers   = parse_answers(answers_text)

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

    # Take-home points → the question-level explanation, shown as the summary note.
    points = [p.strip() for p in (entry.get("key_points") or []) if isinstance(p, str) and p.strip()]
    explanation = "\n".join(f"• {p}" for p in points) or None

    images = [u for u in (entry.get("_image_urls") or []) if u]

    return {
        "question_number": number,
        "question_text": stem,
        "subject": subject,
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
