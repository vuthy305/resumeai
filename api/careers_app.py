"""PUBLIC careers website for candidates.

    uvicorn api.careers_app:app --port 8000        (or: python run.py, which starts both websites)

Isolated from HR on purpose:
  * only 4 endpoints exist here: site config, open jobs, one open job, submit application
  * job data is reduced to what a candidate should see (no applicant counts or scores)
  * no HR pages, no candidate data, no CV downloads, no API docs
  * no AI model is loaded; the HR server analyses new applications in the background
"""
import src  # noqa: F401,I001  (Windows OpenMP fix; harmless here, kept for consistency)
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.common import install_error_handlers, mount_website
from api.routes.applications import create_application, submitted
from src.config import COMPANY_NAME, EMPLOYMENT_TYPES
from src.database.db import get_db, init_db
from src.database.models import Job

PUBLIC_FIELDS = ("id", "title", "department", "description", "required_skills", "preferred_skills",
                 "minimum_experience", "education_requirement", "location", "employment_type", "closing_date")


def public_job(job: Job) -> dict:
    d = {k: getattr(job, k) for k in PUBLIC_FIELDS}
    d["required_skills"], d["preferred_skills"] = job.required_skills or [], job.preferred_skills or []
    d["status"] = job.status
    d["created_at"] = job.created_at.isoformat() if job.created_at else None
    return d


router = APIRouter(tags=["careers"])


@router.get("/site-config")
def site_config():
    return {"company_name": COMPANY_NAME, "employment_types": EMPLOYMENT_TYPES}


@router.get("/jobs")
def open_jobs(db: Session = Depends(get_db)):
    """Open positions only, whatever filter the browser asks for."""
    return [public_job(j) for j in db.scalars(select(Job).where(Job.status == "Open").order_by(Job.id.desc()))]


@router.get("/jobs/{job_id}")
def open_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job or job.status != "Open":
        raise HTTPException(404, "This position is closed or does not exist.")
    return public_job(job)


@router.post("/applications", status_code=201)
async def apply(job_id: int = Form(...), full_name: str = Form(..., min_length=2),
                email: str = Form(..., min_length=5), phone: str = Form(""),
                file: UploadFile = File(...), db: Session = Depends(get_db)):
    # any auto_analyze field the browser sends is ignored: analysis happens on the HR server
    return submitted(*await create_application(db, job_id, full_name, email, phone, file))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title=f"{COMPANY_NAME} Careers", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
install_error_handlers(app)
app.include_router(router)
mount_website(app, "careers")