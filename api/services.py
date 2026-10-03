"""Model manager (singleton, loaded once at startup) + application analysis service."""
import io
import re
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image

from src.config import (ALLOWED_EXTENSIONS, BEST_MODEL_DIR, IMAGE_EXTENSIONS, MAX_UPLOAD_MB, SKIP_MODEL,
                        UPLOAD_DIR)
from src.database.models import Application, Candidate
from src.matching.matcher import match_candidate
from src.pipeline import CVPipeline, PipelineError
from src.utils.hardware import get_hardware_info
from src.utils.logging_utils import get_logger

log = get_logger("services")


class ModelManager:
    """Loads PaddleOCR + LayoutLMv3 exactly once and keeps them in memory."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pipeline = None
            cls._instance.device = "cpu"
            cls._instance.notes = []
        return cls._instance

    def load(self):
        if self.pipeline is not None:
            return self
        self.device = get_hardware_info()["device"]
        ocr, extractor = None, None
        if SKIP_MODEL:
            self.notes.append("RESUMEAI_SKIP_MODEL=1: using rule-based baseline and PDF text layer.")
        else:
            try:
                from src.document_processing.ocr import OCRProcessor
                ocr = OCRProcessor()
            except Exception as e:
                log.warning(f"PaddleOCR unavailable: {e}")
                self.notes.append("PaddleOCR unavailable: image CVs cannot be read; PDFs use their text layer.")
            if (Path(BEST_MODEL_DIR) / "config.json").exists():
                try:
                    from src.extraction.inference import LayoutLMv3Extractor
                    extractor = LayoutLMv3Extractor(BEST_MODEL_DIR, device=self.device)
                except Exception as e:
                    log.exception("LayoutLMv3 failed to load")
                    self.notes.append(f"LayoutLMv3 failed to load ({type(e).__name__}); using rule-based baseline.")
            else:
                self.notes.append(f"No trained checkpoint at {BEST_MODEL_DIR}; using rule-based baseline. "
                                  "Run scripts/train_layoutlmv3.py.")
        self.pipeline = CVPipeline(ocr=ocr, extractor=extractor)
        for n in self.notes:
            log.info(n)
        return self

    @property
    def model_loaded(self) -> bool:
        return bool(self.pipeline and self.pipeline.extractor)

    @property
    def ocr_loaded(self) -> bool:
        return bool(self.pipeline and self.pipeline.ocr)


def get_manager() -> ModelManager:
    return ModelManager().load()


# ----------------------------- uploads -----------------------------
def _safe_name(name: str) -> str:
    base = Path(name or "cv").name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", base)[:150] or "cv"


def validate_bytes(data: bytes, ext: str):
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB).")
    if ext == ".pdf" and not data.lstrip()[:5].startswith(b"%PDF"):
        raise HTTPException(400, "The file is not a valid PDF.")
    if ext == ".docx" and not data.startswith(b"PK"):
        raise HTTPException(400, "The file is not a valid DOCX document.")
    if ext in IMAGE_EXTENSIONS:
        try:
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            raise HTTPException(400, "The image file is corrupted or unsupported.")


async def read_upload(file: UploadFile) -> tuple[bytes, str, str]:
    original = _safe_name(file.filename)
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file format '{ext}'. Allowed: PDF, DOCX, PNG, JPG.")
    data = await file.read()
    validate_bytes(data, ext)
    return data, original, ext


def store_upload(data: bytes, ext: str) -> str:
    stored = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / stored).write_bytes(data)
    return stored


def resolve_cv_path(stored_name: str) -> Path:
    """Only files directly inside UPLOAD_DIR can be served (no path traversal)."""
    root = UPLOAD_DIR.resolve()
    path = (root / Path(stored_name).name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(404, "CV file not found.")
    return path


# ----------------------------- analysis -----------------------------
def process_candidate(db, cand: Candidate, force: bool = False) -> dict:
    if cand.extracted_profile_json and not force:
        return cand.extracted_profile_json
    result = get_manager().pipeline.process_file(resolve_cv_path(cand.cv_path))
    stored = {**result["profile"], "_meta": {k: result[k] for k in (
        "extraction_method", "ocr_engine", "pages_processed", "field_sources", "processing_seconds", "evidence",
        "core_entities")}}
    cand.extracted_text = result["extracted_text"]
    cand.extracted_profile_json = stored
    cand.processed_at = datetime.now()
    db.commit()
    return stored


def analyze_application(db, app: Application, force: bool = False) -> Application:
    try:
        profile = process_candidate(db, app.candidate, force)
        m = match_candidate(profile, app.job.to_matching_dict())
        app.match_score = m["match_score"]
        app.matched_skills = m["required"]["matched"]
        app.missing_skills = m["required"]["missing"]
        app.experience_match = m["experience"]["met"]
        app.education_match = m["education"]["met"]
        app.match_details = m
        app.analysis_status, app.analysis_error = "done", None
    except (PipelineError, ValueError, HTTPException) as e:
        msg = getattr(e, "detail", None) or str(e)
        log.warning(f"Analysis failed for application {app.id}: {msg}")
        app.analysis_status, app.analysis_error = "failed", str(msg)[:500]
    except Exception:
        log.exception(f"Unexpected analysis error for application {app.id}")
        app.analysis_status, app.analysis_error = "failed", "Internal processing error (see server logs)."
    db.commit()
    return app


def analyze_in_background(app_id: int):
    from src.database.db import SessionLocal
    db = SessionLocal()
    try:
        app = db.get(Application, app_id)
        if app:
            analyze_application(db, app)
    finally:
        db.close()


def application_row(app: Application) -> dict:
    cand, prof = app.candidate, app.candidate.extracted_profile_json or {}
    details = app.match_details or {}
    req_total = len(app.job.required_skills or [])
    return {
        "application_id": app.id, "job_id": app.job_id, "candidate_id": cand.id,
        "candidate_name": cand.name, "email": cand.email, "match_score": app.match_score,
        "matched_skills": app.matched_skills or [], "missing_skills": app.missing_skills or [],
        "required_matched": f"{len(app.matched_skills or [])}/{req_total}",
        "experience_years": prof.get("total_experience_years"),
        "experience_match": app.experience_match, "education_match": app.education_match,
        "candidate_skills": prof.get("skills", []),
        "preferred_matched": (details.get("preferred") or {}).get("matched", []),
        "status": app.status, "analysis_status": app.analysis_status, "analysis_error": app.analysis_error,
        "created_at": app.created_at.isoformat() if app.created_at else None,
    }
