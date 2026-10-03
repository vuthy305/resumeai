from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.routes.jobs import get_job_or_404
from api.schemas import StatusUpdate
from api.services import (analyze_application, analyze_in_background, application_row, read_upload,
                          resolve_cv_path, store_upload)
from src.config import APPLICATION_STATUSES
from src.database.db import get_db
from src.database.models import Application, Candidate, Job

router = APIRouter(tags=["applications"])
MEDIA = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


async def create_application(db: Session, job_id: int, full_name: str, email: str, phone: str,
                             file: UploadFile) -> tuple[Application, Candidate, Job]:
    """Validate and store one application. Shared by the careers site and the HR API."""
    job = get_job_or_404(db, job_id)
    if job.status != "Open":
        raise HTTPException(400, "This job is no longer accepting applications.")
    if "@" not in email:
        raise HTTPException(422, "Please provide a valid email address.")
    data, original, ext = await read_upload(file)
    cand = db.scalar(select(Candidate).where(Candidate.email == email.strip().lower()))
    if cand and db.scalar(select(Application).where(Application.job_id == job_id,
                                                    Application.candidate_id == cand.id)):
        raise HTTPException(409, "You have already applied for this job.")
    stored = store_upload(data, ext)
    if cand:  # same person, new CV: update and invalidate old analyses
        cand.name, cand.phone, cand.cv_filename, cand.cv_path = full_name.strip(), phone.strip(), original, stored
        cand.extracted_profile_json, cand.extracted_text = None, None
        for a in cand.applications:
            a.match_score, a.analysis_status, a.match_details = None, "pending", None
    else:
        cand = Candidate(name=full_name.strip(), email=email.strip().lower(), phone=phone.strip() or None,
                         cv_filename=original, cv_path=stored)
        db.add(cand)
        db.flush()
    app = Application(job_id=job_id, candidate_id=cand.id)
    db.add(app)
    db.commit()
    return app, cand, job


def submitted(app, cand, job) -> dict:
    return {"application_id": app.id, "candidate_id": cand.id, "job_id": job.id, "job_title": job.title,
            "status": app.status, "message": "Application submitted successfully."}


@router.post("/applications", status_code=201)
async def submit_application(background: BackgroundTasks,
                             job_id: int = Form(...), full_name: str = Form(..., min_length=2),
                             email: str = Form(..., min_length=5), phone: str = Form(""),
                             auto_analyze: bool = Form(False), file: UploadFile = File(...),
                             db: Session = Depends(get_db)):
    app, cand, job = await create_application(db, job_id, full_name, email, phone, file)
    if auto_analyze:
        background.add_task(analyze_in_background, app.id)
    return submitted(app, cand, job)


def _get_app(db, application_id) -> Application:
    app = db.get(Application, application_id)
    if not app:
        raise HTTPException(404, f"Application {application_id} not found.")
    return app


@router.post("/applications/{application_id}/analyze")
def analyze(application_id: int, force: bool = False, db: Session = Depends(get_db)):
    app = analyze_application(db, _get_app(db, application_id), force)
    return {**application_row(app), "match_details": app.match_details}


@router.get("/applications/{application_id}")
def get_application(application_id: int, db: Session = Depends(get_db)):
    app = _get_app(db, application_id)
    return {**application_row(app), "match_details": app.match_details,
            "job": {"id": app.job.id, "title": app.job.title}}


@router.patch("/applications/{application_id}/status")
def update_status(application_id: int, payload: StatusUpdate, db: Session = Depends(get_db)):
    if payload.status not in APPLICATION_STATUSES:
        raise HTTPException(422, f"Status must be one of {APPLICATION_STATUSES}")
    app = _get_app(db, application_id)
    app.status = payload.status
    db.commit()
    return {"application_id": app.id, "status": app.status}


@router.get("/jobs/{job_id}/applications")
def job_applications(job_id: int, db: Session = Depends(get_db)):
    job = get_job_or_404(db, job_id)
    return [application_row(a) for a in job.applications]


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: int, db: Session = Depends(get_db)):
    cand = db.get(Candidate, candidate_id)
    if not cand:
        raise HTTPException(404, f"Candidate {candidate_id} not found.")
    return {"id": cand.id, "name": cand.name, "email": cand.email, "phone": cand.phone,
            "cv_filename": cand.cv_filename, "cv_url": f"/candidates/{cand.id}/cv",
            "profile": cand.extracted_profile_json, "processed_at": cand.processed_at,
            "applications": [{"application_id": a.id, "job_id": a.job_id, "job_title": a.job.title,
                              "match_score": a.match_score, "status": a.status} for a in cand.applications]}


@router.get("/candidates/{candidate_id}/cv")
def get_candidate_cv(candidate_id: int, download: bool = False, db: Session = Depends(get_db)):
    cand = db.get(Candidate, candidate_id)
    if not cand:
        raise HTTPException(404, f"Candidate {candidate_id} not found.")
    path = resolve_cv_path(cand.cv_path)
    return FileResponse(path, media_type=MEDIA.get(path.suffix.lower(), "application/octet-stream"),
                        filename=cand.cv_filename,
                        content_disposition_type="attachment" if download else "inline")
