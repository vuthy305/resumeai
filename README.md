# ResumeAI — AI-Powered CV Information Extraction and HR Candidate Matching Dashboard

Academic MVP. Candidates apply to a job and upload a CV. HR sees only the applicants for the selected job. The
system extracts a structured profile from each CV, scores it against the job's requirements with a transparent
formula, and ranks the applicants.

> ResumeAI provides AI-assisted candidate matching for recruiter review. The score is not a hiring decision.
> HR must review the original CV before making recruitment decisions.

## 1. Research boundary

```
MAIN AI MODEL     LayoutLMv3 (microsoft/layoutlmv3-base, LayoutLMv3ForTokenClassification, fully fine-tuned)
                    -> NAME, EMAIL, PHONE, ADDRESS, SECTIONHEADER   (supervised by Kunling/layoutlm_resume_data)
BASELINE          PaddleOCR + Regex / Keyword / Rules   (same 5 entities)
SUPPLEMENTARY     Rule-based parser: skills, education, degree, job titles, companies, experience,
                  certifications, languages.  NOT learned by LayoutLMv3 (the dataset has no labels for them).
MATCHING          Structured CV + job requirements -> transparent weighted formula -> "AI Match Score"
```

The Kaggle Resume Dataset (snehaanbhawal) is used only for demo and OCR testing, never as training data.

## 2. Installation (Windows, Python 3.10 or 3.11)

```powershell
cd resumeai
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) PyTorch with CUDA 12.1 (special install; not part of requirements.txt)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
# 2) everything else
pip install -r requirements.txt
```

Notes:
- PaddleOCR uses the **CPU** build of paddlepaddle on purpose, so the 4 GB of VRAM stays free for LayoutLMv3. You can
  set `$env:RESUMEAI_OCR_GPU="1"` only if you install `paddlepaddle-gpu` yourself. This is not recommended on 4 GB.
- PaddleOCR 2.x needs `numpy<2`, which is pinned.
- `scripts/download_dataset.py` stores the dataset permanently in `data/raw/layoutlm_resume_data/` (`files/` = exact
  repository files, `arrow/` = ready-to-load copy). `prepare_dataset.py` loads it from disk when it is there, so
  no internet is needed for the dataset after that.
- The first run downloads `microsoft/layoutlmv3-base` (~500 MB), the PaddleOCR models (~15 MB) and the dataset.

## 3. Run order (exact commands)

Run everything from the project root. Each command is one line, so it works the same in PowerShell and bash.

| # | Step | Command | Output |
|---|------|---------|--------|
| 1 | Environment | `python scripts/check_environment.py` | versions, CUDA, GPU, VRAM, import + SQLite checks |
| 2a | Download dataset (once) | `python scripts/download_dataset.py` | local copy in `data/raw/layoutlm_resume_data/` + browsable `view/index.html` |
| 2 | Dataset | `python scripts/prepare_dataset.py` | `data/processed/resume_ner`, `label_map.json`, `results/dataset_stats.json` |
| 3 | Baseline eval | `python scripts/evaluate_baseline.py --mode tokens` | `results/baseline_results.csv` |
| 3b | Baseline incl. real OCR | `python scripts/evaluate_baseline.py --mode ocr` | string-level OCR results |
| 4 | Fine-tune | `python scripts/train_layoutlmv3.py --epochs 5 --batch-size 1 --gradient-accumulation 8 --max-length 256 --learning-rate 5e-5 --fp16 --gradient-checkpointing` | `models/layoutlmv3/best`, `results/parameter_report.txt`, `results/training_log.json` |
| 5 | Evaluate | `python scripts/evaluate_model.py --mode tokens` | `layoutlmv3_results.csv`, `extraction_comparison.csv`, `comparison.csv`, `entity_f1.png`, `model_comparison.png` |
| 6 | Ablation | `python scripts/run_ablation.py --epochs 5` | `models/layoutlmv3_text_only/best`, `ablation_results.csv`, `ablation.png` |
| 6b | Matching vs baselines | `python scripts/evaluate_matching.py` (add `--embeddings` for the deep-learning baseline) | `matching_results.csv`, `matching_comparison.png`, `matching_predictions.csv` |
| 7 | Demo data | `python scripts/seed_demo.py --reset` | 3 jobs, 12 synthetic candidates, 15 applications |
| 8 | Start both websites | `python run.py` | Careers http://localhost:8000 · HR http://localhost:8001 |
| 9 | Tests | `pytest -q` | |

### Two separate websites

| Website | Address | Who | What it can do |
|---|---|---|---|
| Careers site (public) | http://localhost:8000 | Candidates | See open jobs, apply with a CV. Nothing else: no HR pages, no applicant data, no API docs |
| HR portal (internal) | http://localhost:8001 | HR staff | Create/edit/close jobs, dashboard, AI analysis, ranking, CVs, status. API docs at `/docs` |

They are two separate servers (`api/careers_app.py` and `api/main.py`) that share only the database and the
uploads folder, like a public website and an internal system in a real bank. There is no link from the careers site
to HR. Only the HR server loads the AI model; it analyses new applications automatically every 10 seconds
(`RESUMEAI_AUTO_ANALYZE_SECONDS`, 0 = off). `python run.py` starts both; or use two terminals:
`uvicorn api.careers_app:app --port 8000` and `uvicorn api.main:app --port 8001`.

The HR portal has no login in this MVP; in production it would sit on the bank's internal network behind
staff sign-in.

## 4. Two-minute demo

1. **HR portal (http://localhost:8001) → Jobs → New job**: create a job (title, required and preferred skills with suggestions from the skill
   dictionary, minimum experience, minimum education, deadline). Click **Publish job**.
2. **Careers site** (http://localhost:8000): the new job appears immediately. Open a job, fill in name, email and phone, attach a CV (PDF,
   DOCX, JPG or PNG), tick consent, click **Submit application**. The candidate gets a reference number.
   The HR server analyses the CV automatically within about 10 seconds.
3. **HR portal → Dashboard**: counts of open jobs, applications, new in 24 h, waiting for analysis, shortlisted,
   interview, per-job table and latest applications. It refreshes every 15 seconds.
4. **HR portal → a job**: ranked applicants sorted by AI Match Score, with filters (name, minimum score, status,
   required skill, minimum experience). **Analyze waiting CVs** / **Re-analyze all** run the AI pipeline.
5. Click an applicant: **Match** tab (score breakdown by component, ✓ matched and ✗ missing skills, the formula),
   **Extracted profile** tab (what LayoutLMv3 extracted vs. what the rule-based parser extracted), **Original CV** tab.
6. Change the status: **Mark under review / Shortlist / Invite to interview / Reject**.
7. **Close job**: it disappears from the careers site. Editing a job's requirements marks its applications for
   re-analysis so scores never use outdated requirements.

## 5. Matching formula

```
AI Match Score = required_skill_score*0.50 + experience_score*0.25 + education_score*0.15 + preferred_skill_score*0.10
```

Each component is 0–100. Skills are normalised with `data/skill_dictionary.json`. The score is a transparent
job-fit indicator. It is **not** a hiring probability.

Protected attributes (gender, age, religion, race, marital status, health, disability, politics, family status,
photo) are never extracted or used.

## 6. Fallbacks (always reported, never hidden)

| Situation | Behaviour | Shown as |
|---|---|---|
| No trained checkpoint in `models/layoutlmv3/best` | Core entities come from the rule baseline | `/health` → `extraction_method: rule_based_baseline` |
| PaddleOCR not installed / fails on a PDF | The PDF's embedded text layer is used (real PyMuPDF word boxes) | `ocr_engine: pdf_text_layer` |
| Image CV with no OCR available | Analysis fails with a clear message | application `analysis_status: failed` |
| DOCX | Text is extracted with python-docx, then the rules run (no layout image) | `extraction_method: rule_based_text (docx)` |
| No CUDA | Everything runs on the CPU | `device: cpu` |

`RESUMEAI_SKIP_MODEL=1` forces the fallback mode, which is used by the tests.

## 7. API

`GET /health` · `GET /jobs` (`?status=Open`) · `GET /jobs/{id}` · `POST /jobs` · `PATCH /jobs/{id}` · `GET /skills` ·
`GET /site-config` · `GET /applications/recent` · `POST /extract` · `POST /applications` ·
`POST /applications/{id}/analyze` · `GET /applications/{id}` · `PATCH /applications/{id}/status` ·
`GET /jobs/{id}/applications` · `GET /jobs/{id}/ranking` · `POST /jobs/{id}/analyze` · `GET /candidates/{id}` ·
`GET /candidates/{id}/cv` · `GET /stats`

Uploads are validated by extension, size (10 MB) and file signature. They are stored under random UUID names, and
CVs are served only by candidate id, never by path. Errors return JSON messages, and stack traces go to the server
log only.

## 8. Configuration (environment variables)

`RESUMEAI_DB_URL` (default `sqlite:///resumeai.db`), `RESUMEAI_UPLOAD_DIR`, `RESUMEAI_COMPANY_NAME`, `RESUMEAI_CAREERS_PORT`, `RESUMEAI_HR_PORT`, `RESUMEAI_AUTO_ANALYZE_SECONDS`,
`RESUMEAI_SKIP_MODEL`, `RESUMEAI_OCR_GPU`.
