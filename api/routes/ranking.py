from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.routes.jobs import get_job_or_404
from api.services import analyze_application, application_row
from src.database.db import get_db
from src.matching.ranking import rank_applications

router = APIRouter(tags=["ranking"])


@router.get("/jobs/{job_id}/ranking")
def ranking(job_id: int, db: Session = Depends(get_db)):
    job = get_job_or_404(db, job_id)
    return rank_applications([application_row(a) for a in job.applications])


@router.post("/jobs/{job_id}/analyze")
def analyze_job(job_id: int, force: bool = False, db: Session = Depends(get_db)):
    """Analyse every pending (or all, if force) application of one job, sequentially."""
    job = get_job_or_404(db, job_id)
    done = failed = 0
    for app in job.applications:
        if force or app.analysis_status != "done":
            analyze_application(db, app, force)
            done += app.analysis_status == "done"
            failed += app.analysis_status == "failed"
    return {"job_id": job_id, "analyzed": done, "failed": failed,
            "ranking": rank_applications([application_row(a) for a in job.applications])}
