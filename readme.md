source venv/bin/activate

uvicorn app.main:app --reload

---

## Endpoints

### POST /import  — recall questions (text PDF pair)

Two PDFs (questions + answers), parsed with regex. See app/main.py.

### POST /import/qbank  — screenshot-export MCQ PDF → qbank import batch

One PDF where every page is a screenshot of another quiz app (no selectable
text). Each answered page carries a question + its per-option explanations and a
marked correct answer; Claude Haiku vision reads each page and the questions are
pushed to the backend as a `qbank` import batch.

Needs `ANTHROPIC_API_KEY` in .env. ~$0.30–0.50 per ~200 questions.

Two source layouts are handled:

- **Interleaved** (AMC "209 MCQ"): page 1 = Q1 unanswered, page 2 = Q1 answered,
  page 3 = Q2 unanswered, … Default `mode=answered` reads the even pages.
  ```
  curl -F 'pdf=@"AMC Free 209 MCQ.pdf"' -F 'title=AMC Free 209 MCQ' \
       http://localhost:8000/import/qbank
  ```

- **Blocked** (eMedici): pages 1–N are the questions, pages N+1–2N are the same
  questions answered. Pass the answer block as an explicit range:
  ```
  curl -F 'pdf=@"eMedici mock 2025.pdf"' -F 'title=eMedici Mock 2025' \
       -F 'pages=121-240' http://localhost:8000/import/qbank
  ```

`pages` accepts integers and ranges (`"41,42,43"`, `"121-240"`). Use a small
list to backfill a handful of questions the first pass missed, at a few cents.

**Images.** Clinical figures (X-rays, scans, ECGs, clinical photos) are detected,
cropped tight, and uploaded to the private S3 bucket via the backend
(`POST /api/admin/import-batches/images`). Students read them back through
`GET /api/images/question`. `mode=all` reads every page if a PDF's layout is
irregular.

Returns immediately with `{ batch_id }`. Extraction runs in the background
(~10 min for ~200 questions) — watch the batch flip from Processing to
Completed/Failed on the admin **Import Batches** page, then set it visible / free
and approve it there.
