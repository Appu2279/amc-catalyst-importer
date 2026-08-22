from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import tempfile
import os
import uuid
import pdfplumber
import fitz  # PyMuPDF
import httpx
import re

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

        answers[question_number] = {
            "correct_answer":      answer_letter,
            "explanation":         explanation,
            "option_explanations": option_explanations if option_explanations else None,
        }

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
# A second paper layout, with no bracketed metadata:
#
#   Question 1
#   <stem paragraphs>
#   Which one of the following is the most appropriate management?
#   A. First option B. Second option C. Third option
#
# and answers as:
#
#   Q1 — Answer: C. Third option
#   <explanation paragraphs>
#   ● A. First option — wrong; ...
#
# Note the options run inline and wrap mid-option across lines, so they cannot
# be split on newlines the way the labelled format is.

_PROSE_QUESTION_RE = re.compile(r'(?m)^Question\s+(\d+)\s*$')
# Consumes the rest of the answer line: it repeats the correct option's text
# ("Answer: C. Maintain sertraline at 50 mg daily"), which would otherwise be
# read as the opening words of the explanation.
_PROSE_ANSWER_RE = re.compile(r'(?m)^Q(\d+)\s*[—–-]\s*Answer:\s*([A-F])[.:]?[^\n]*')


def _split_inline_options(chunk):
    """Separate an inline 'A. x B. y C. z' run from the stem before it.

    Only a run that starts at A and steps forward one letter at a time counts,
    so an 'A.' inside the stem cannot start the options early — the sequence
    has to continue with B for anything to be taken as an option.
    """
    flat = ' '.join(chunk.split())

    markers = []
    for match in re.finditer(r'(?:(?<=\s)|^)([A-F])\.\s', flat):
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
            "question_number": int(match.group(1)),
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
    answers = {}
    matches = list(_PROSE_ANSWER_RE.finditer(text))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]

        # The per-option notes are bulleted; everything before the first bullet
        # is the explanation of the correct answer.
        bullet_split = re.split(r'[●•]', block, maxsplit=1)
        explanation = clean_text(' '.join(bullet_split[0].split()))

        option_explanations = {}
        for bullet in re.findall(r'[●•]\s*([A-F])\.\s*([^●•]+)', block):
            option_explanations[bullet[0]] = clean_text(' '.join(bullet[1].split()))

        answers[int(match.group(1))] = {
            "correct_answer":      match.group(2).upper(),
            "explanation":         explanation or None,
            "option_explanations": option_explanations,
        }

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
    """
    merged = []

    for q in questions:
        q_no        = q["question_number"]
        answer_data = answers.get(q_no, {})

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
