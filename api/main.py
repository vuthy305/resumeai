"""INTERNAL HR portal + full ResumeAI API (AI model lives here).

    uvicorn api.main:app --port 8001        (or: python run.py, which starts both websites)

Opening http://localhost:8001 shows the HR portal; http://localhost:8001/docs shows the API.
New applications from the careers site are analysed automatically in the background.
"""
import src  # noqa: F401,I001  (must be first: sets the Windows OpenMP fix before torch/paddle load)
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import select

from api.common import install_error_handlers, mount_website
from api.routes import applications, extraction, jobs, ranking
from api.services import analyze_application, get_manager
from src.config import AUTO_ANALYZE_SECONDS
from src.database.db import SessionLocal, init_db
from src.database.models import Application
from src.utils.logging_utils import get_logger

log = get_logger("api")


def analyze_pending_once() -> int:
    """Analyse every application still waiting (oldest first). Returns how many were processed."""
    db = SessionLocal()
    try:
        apps = db.scalars(select(Application).where(Application.analysis_status == "pending")
                          .order_by(Application.id)).all()
        for app in apps:
            if app.analysis_status == "pending":  # HR may have analysed it meanwhile
                analyze_application(db, app)
        return len(apps)
    finally:
        db.close()


async def auto_analyze_loop():
    while True:
        await asyncio.sleep(AUTO_ANALYZE_SECONDS)
        try:
            n = await asyncio.to_thread(analyze_pending_once)
            if n:
                log.info(f"Auto-analysis: processed {n} new application(s)")
        except Exception:
            log.exception("Auto-analysis loop error (will retry)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_manager()  # load OCR + LayoutLMv3 exactly once
    task = asyncio.create_task(auto_analyze_loop()) if AUTO_ANALYZE_SECONDS > 0 else None
    log.info("ResumeAI HR server ready" + (f" (auto-analysis every {AUTO_ANALYZE_SECONDS:g}s)" if task else ""))
    yield
    if task:
        task.cancel()


app = FastAPI(title="ResumeAI HR API", version="1.1",
              description="Internal HR API: AI-assisted CV extraction and candidate matching for recruiter review. "
                          "The AI Match Score is not a hiring decision.", lifespan=lifespan)
install_error_handlers(app)


@app.get("/health")
def health():
    m = get_manager()
    return {"status": "ok", "model_loaded": m.model_loaded, "device": m.device, "ocr_loaded": m.ocr_loaded,
            "extraction_method": m.pipeline.core_method, "notes": m.notes}


for r in (jobs.router, applications.router, extraction.router, ranking.router):
    app.include_router(r)

mount_website(app, "hr")  # HR portal at "/" (after the API routes)
