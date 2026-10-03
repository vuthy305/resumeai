import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from src.database.db import check_connection, init_db, make_engine
from src.database.models import Application, Candidate, Job


def test_database_crud_in_memory():
    engine = make_engine("sqlite://")
    init_db(engine)
    assert check_connection(engine)
    db = sessionmaker(bind=engine)()
    job = Job(title="Data Analyst", required_skills=["SQL"], preferred_skills=[], minimum_experience=1)
    cand = Candidate(name="Test Person", email="t@example.com", cv_filename="cv.pdf", cv_path="x.pdf")
    db.add_all([job, cand]); db.flush()
    db.add(Application(job_id=job.id, candidate_id=cand.id)); db.commit()
    app = db.query(Application).one()
    assert app.job.title == "Data Analyst" and app.status == "Submitted" and app.match_score is None


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def job_id(client):
    r = client.post("/jobs", json={"title": "Backend Developer", "department": "IT",
                                   "required_skills": ["Python", "PostgreSQL", "REST API"],
                                   "preferred_skills": ["Docker", "FastAPI"], "minimum_experience_years": 2,
                                   "education_requirement": "Bachelor's degree in Computer Science"})
    assert r.status_code == 201
    return r.json()["id"]


def _apply(client, job_id, pdf_bytes, email="dara.vann@example.com", name="cv.pdf"):
    return client.post("/applications", data={"job_id": job_id, "full_name": "Dara Vann", "email": email,
                                              "phone": "012345678"},
                       files={"file": (name, pdf_bytes, "application/pdf")})


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is False and h["extraction_method"] == "rule_based_baseline"


def test_full_application_flow(client, job_id, sample_pdf):
    pdf = sample_pdf.read_bytes()
    r = _apply(client, job_id, pdf)
    assert r.status_code == 201, r.text
    app_id, cand_id = r.json()["application_id"], r.json()["candidate_id"]
    assert _apply(client, job_id, pdf).status_code == 409                   # duplicate application
    assert client.get("/jobs").json()[0]["applicant_count"] == 1
    lst = client.get(f"/jobs/{job_id}/applications").json()
    assert lst[0]["analysis_status"] == "pending"

    a = client.post(f"/applications/{app_id}/analyze").json()
    assert a["analysis_status"] == "done" and a["match_score"] == 100.0
    assert a["matched_skills"] == ["Python", "PostgreSQL", "REST API"] and a["missing_skills"] == []

    rank = client.get(f"/jobs/{job_id}/ranking").json()
    assert rank[0]["rank"] == 1 and rank[0]["candidate_id"] == cand_id

    cand = client.get(f"/candidates/{cand_id}").json()
    assert cand["profile"]["name"] == "Dara Vann"
    cv = client.get(f"/candidates/{cand_id}/cv")
    assert cv.status_code == 200 and cv.content == pdf and cv.headers["content-type"] == "application/pdf"

    assert client.patch(f"/applications/{app_id}/status", json={"status": "Interview"}).json()["status"] == "Interview"
    assert client.patch(f"/applications/{app_id}/status", json={"status": "Hired!"}).status_code == 422
    assert client.get("/stats").json()["by_status"]["Interview"] == 1


def test_upload_validation(client, job_id):
    assert _apply(client, job_id, b"hello", email="a@b.com", name="cv.exe").status_code == 415
    assert _apply(client, job_id, b"not a pdf", email="a@b.com").status_code == 400
    assert _apply(client, job_id, b"", email="a@b.com").status_code == 400
    assert _apply(client, 9999, b"%PDF-1.4", email="a@b.com").status_code == 404


def test_corrupted_pdf_analysis_fails_gracefully(client, job_id):
    r = _apply(client, job_id, b"%PDF-1.4 broken content", email="broken@example.com")
    assert r.status_code == 201
    a = client.post(f"/applications/{r.json()['application_id']}/analyze").json()
    assert a["analysis_status"] == "failed" and "Traceback" not in (a["analysis_error"] or "")
    rank = client.get(f"/jobs/{job_id}/ranking").json()
    assert rank[-1]["rank"] is None                                      # failed ones listed last


def test_extract_endpoint_and_missing_resources(client, sample_pdf):
    r = client.post("/extract", files={"file": ("cv.pdf", sample_pdf.read_bytes(), "application/pdf")})
    assert r.status_code == 200 and r.json()["profile"]["email"] == "dara.vann@example.com"
    assert client.get("/candidates/12345").status_code == 404
    assert client.get("/jobs/12345/ranking").status_code == 404
    assert client.post("/applications/12345/analyze").status_code == 404


def test_cv_path_traversal_is_blocked(client):
    from api.services import resolve_cv_path
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        resolve_cv_path("../../etc/passwd")
