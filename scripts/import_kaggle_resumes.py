"""Optional: add a few real PDF resumes from the Kaggle 'Resume Dataset' (Snehaan Bhawal, CC0)
as applicants for a job, for realistic OCR/document testing. Resumes there are anonymised,
so candidates are labelled 'Kaggle Resume <id>' with placeholder emails.
    python scripts/import_kaggle_resumes.py --src data/raw/kaggle/data/data --category INFORMATION-TECHNOLOGY --job "Backend Developer" --n 5
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import UPLOAD_DIR  # noqa: E402
from src.database.db import SessionLocal, init_db  # noqa: E402
from src.database.models import Application, Candidate, Job  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with <CATEGORY>/<id>.pdf")
    ap.add_argument("--category", default="INFORMATION-TECHNOLOGY")
    ap.add_argument("--job", default="Backend Developer")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    init_db()
    db = SessionLocal()
    job = db.query(Job).filter(Job.title == a.job).first()
    if not job:
        raise SystemExit(f"Job '{a.job}' not found. Run scripts/seed_demo.py first.")
    pdfs = sorted((Path(a.src) / a.category).glob("*.pdf"))[: a.n]
    if not pdfs:
        raise SystemExit(f"No PDFs in {Path(a.src) / a.category}")
    for p in pdfs:
        email = f"kaggle.{p.stem}@example.invalid"
        if db.query(Candidate).filter(Candidate.email == email).first():
            continue
        stored = f"kaggle_{p.name}"
        shutil.copy(p, UPLOAD_DIR / stored)
        c = Candidate(name=f"Kaggle Resume {p.stem}", email=email, cv_filename=p.name, cv_path=stored)
        db.add(c); db.flush()
        db.add(Application(job_id=job.id, candidate_id=c.id))
    db.commit()
    print(f"Added {len(pdfs)} Kaggle resumes as applicants for {a.job}.")


if __name__ == "__main__":
    main()
