from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.schemas import JobCreate, JobUpdate
from src.config import APPLICATION_STATUSES, CAREERS_URL, COMPANY_NAME, EMPLOYMENT_TYPES, JOB_STATUSES
from src.database.db import get_db
from src.database.models import Application, Candidate, Job
from src.matching.matcher import LEVEL_NAMES
from src.utils.skills import get_normalizer

router = APIRouter(tags=["jobs"])


def job_dict(job: Job, summary: dict | None = None) -> dict:
    s = summary or {}
    return {"id": job.id, "title": job.title, "department": job.department, "description": job.description,
            "required_skills": job.required_skills or [], "preferred_skills": job.preferred_skills or [],
            "minimum_experience": job.minimum_experience, "education_requirement": job.education_requirement,
            "location": job.location or "", "employment_type": job.employment_type or "",
            "closing_date": job.closing_date, "status": job.status,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "applicant_count": s.get("applicants", 0), "analyzed_count": s.get("analyzed", 0),
            "pending_count": s.get("pending", 0), "new_today": s.get("new_today", 0),
            "top_score": s.get("top_score"), "status_counts": s.get("status_counts", {})}


def job_summaries(db: Session) -> dict[int, dict]:
    since = datetime.now() - timedelta(hours=24)
    out: dict[int, dict] = {}
    rows = db.execute(select(Application.job_id, Application.status, Application.analysis_status,
                             Application.match_score, Application.created_at)).all()
    for job_id, status, analysis, score, created in rows:
        s = out.setdefault(job_id, {"applicants": 0, "analyzed": 0, "pending": 0, "new_today": 0,
                                    "top_score": None, "status_counts": {}})
        s["applicants"] += 1
        s["analyzed"] += analysis == "done"
        s["pending"] += analysis == "pending"
        s["new_today"] += bool(created and created >= since)
        s["status_counts"][status] = s["status_counts"].get(status, 0) + 1
        if score is not None and (s["top_score"] is None or score > s["top_score"]):
            s["top_score"] = score
    return out


def get_job_or_404(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")
    return job


def _clean_skills(skills: list[str]) -> list[str]:
    """Normalise to dictionary names (e.g. 'postgres' -> 'PostgreSQL') so matching works; keep unknown skills."""
    norm, out = get_normalizer(), []
    for sk in skills:
        sk = norm.normalize(sk.strip()) if sk and sk.strip() else ""
        if sk and sk.lower() not in {o.lower() for o in out}:
            out.append(sk)
    return out


def _validate(status=None, employment_type=None):
    if status is not None and status not in JOB_STATUSES:
        raise HTTPException(422, f"Job status must be one of {JOB_STATUSES}")
    if employment_type is not None and employment_type not in EMPLOYMENT_TYPES:
        raise HTTPException(422, f"Employment type must be one of {EMPLOYMENT_TYPES}")


@router.get("/jobs")
def list_jobs(status: str | None = None, db: Session = Depends(get_db)):
    """All jobs (HR portal) or only ?status=Open (careers site)."""
    q = select(Job).order_by(Job.id.desc())
    if status:
        q = q.where(Job.status == status)
    sums = job_summaries(db)
    return [job_dict(j, sums.get(j.id)) for j in db.scalars(q)]


@router.get("/jobs/{job_id}")
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = get_job_or_404(db, job_id)
    return job_dict(job, job_summaries(db).get(job.id))


@router.post("/jobs", status_code=201)
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    _validate(payload.status, payload.employment_type)
    job = Job(title=payload.title.strip(), department=payload.department.strip(),
              description=payload.description.strip(),
              required_skills=_clean_skills(payload.required_skills),
              preferred_skills=_clean_skills(payload.preferred_skills),
              minimum_experience=payload.minimum_experience_years,
              education_requirement=payload.education_requirement.strip(),
              location=payload.location.strip(), employment_type=payload.employment_type,
              closing_date=payload.closing_date or None, status=payload.status)
    db.add(job)
    db.commit()
    return job_dict(job)


@router.patch("/jobs/{job_id}")
def update_job(job_id: int, payload: JobUpdate, db: Session = Depends(get_db)):
    """Edit a job or open/close it. Changing requirements marks its applications for re-analysis."""
    job = get_job_or_404(db, job_id)
    data = payload.model_dump(exclude_unset=True)
    _validate(data.get("status"), data.get("employment_type"))
    requirement_fields = {"required_skills", "preferred_skills", "minimum_experience_years", "education_requirement"}
    before = job.to_matching_dict()
    for key, value in data.items():
        if key in ("required_skills", "preferred_skills"):
            value = _clean_skills(value or [])
        if key == "minimum_experience_years":
            key = "minimum_experience"
        if key == "closing_date":
            value = value or None
        if isinstance(value, str) and key not in ("closing_date",):
            value = value.strip()
        setattr(job, key, value)
    rescore = requirement_fields & data.keys() and job.to_matching_dict() != before
    if rescore:  # scores computed against the old requirements are no longer valid
        for app in job.applications:
            if app.analysis_status == "done":
                app.analysis_status, app.match_score, app.match_details = "pending", None, None
    db.commit()
    return {**job_dict(job, job_summaries(db).get(job.id)), "needs_reanalysis": bool(rescore)}


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    by_status = dict(db.execute(select(Application.status, func.count()).group_by(Application.status)).all())
    count = lambda cond: db.scalar(select(func.count()).select_from(Application).where(cond))  # noqa: E731
    since = datetime.now() - timedelta(hours=24)
    return {"open_jobs": db.scalar(select(func.count()).select_from(Job).where(Job.status == "Open")),
            "total_jobs": db.scalar(select(func.count()).select_from(Job)),
            "applications": sum(by_status.values()),
            "by_status": {s: by_status.get(s, 0) for s in APPLICATION_STATUSES},
            "analyzed": count(Application.analysis_status == "done"),
            "pending_analysis": count(Application.analysis_status == "pending"),
            "failed_analysis": count(Application.analysis_status == "failed"),
            "new_today": count(Application.created_at >= since)}


@router.get("/applications/recent")
def recent_applications(limit: int = 8, db: Session = Depends(get_db)):
    rows = db.execute(select(Application, Candidate.name, Job.title).join(Candidate).join(Job)
                      .order_by(Application.created_at.desc(), Application.id.desc()).limit(min(limit, 50))).all()
    return [{"application_id": a.id, "job_id": a.job_id, "job_title": title, "candidate_name": name,
             "status": a.status, "analysis_status": a.analysis_status, "match_score": a.match_score,
             "created_at": a.created_at.isoformat() if a.created_at else None} for a, name, title in rows]


@router.get("/skills")
def skills():
    """Canonical skill names from data/skill_dictionary.json, used for suggestions when HR writes a job."""
    return sorted(get_normalizer().entries, key=str.lower)


@router.get("/site-config")
def site_config():
    return {"company_name": COMPANY_NAME, "careers_url": CAREERS_URL, "job_statuses": JOB_STATUSES, "employment_types": EMPLOYMENT_TYPES,
            "application_statuses": APPLICATION_STATUSES,
            "education_levels": [LEVEL_NAMES[i] for i in range(1, 6)]}
