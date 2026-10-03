"""The careers site and the HR portal are separate servers; the careers site exposes no HR data."""
from fastapi.testclient import TestClient

from api.careers_app import app as careers_app
from api.main import analyze_pending_once
from api.main import app as hr_app


def _hr_job(hr, **kw):
    body = {"title": "Teller", "required_skills": ["Excel"], "minimum_experience_years": 0, **kw}
    r = hr.post("/jobs", json=body)
    assert r.status_code == 201
    return r.json()["id"]


def test_each_server_serves_its_own_website():
    with TestClient(careers_app) as c, TestClient(hr_app) as hr:
        assert "Careers" in c.get("/").text and "HR portal" not in c.get("/").text
        assert "HR portal" in hr.get("/").text


def test_careers_shows_only_open_jobs_without_hr_fields():
    with TestClient(careers_app) as c, TestClient(hr_app) as hr:
        open_id = _hr_job(hr, title="Open teller")
        closed_id = _hr_job(hr, title="Closed teller", status="Closed")
        jobs = c.get("/jobs?status=Closed").json()  # filter ignored: open jobs only
        ids = {j["id"] for j in jobs}
        assert open_id in ids and closed_id not in ids
        assert all("applicant_count" not in j and "top_score" not in j for j in jobs)
        assert c.get(f"/jobs/{closed_id}").status_code == 404
        assert c.get(f"/jobs/{open_id}").status_code == 200


def test_hr_endpoints_do_not_exist_on_careers_site():
    with TestClient(careers_app) as c:
        for path in ("/stats", "/health", "/jobs/1/ranking", "/jobs/1/applications", "/candidates/1",
                     "/candidates/1/cv", "/applications/1", "/applications/recent", "/skills", "/docs", "/openapi.json"):
            r = c.get(path)
            assert r.status_code == 404 or "text/html" in r.headers.get("content-type", ""), path
            assert "applicant" not in r.text.lower() or "text/html" in r.headers["content-type"]
        assert c.post("/jobs", json={"title": "Hack"}).status_code in (404, 405)
        assert c.patch("/applications/1/status", json={"status": "Interview"}).status_code in (404, 405)


def test_apply_on_careers_then_hr_analyzes(sample_pdf):
    with TestClient(careers_app) as c, TestClient(hr_app) as hr:
        job_id = _hr_job(hr, title="Backend", required_skills=["Python"])
        with open(sample_pdf, "rb") as f:
            r = c.post("/applications", data={"job_id": job_id, "full_name": "Dara Vann", "email": "two@sites.kh",
                                              "auto_analyze": "true"}, files={"file": ("cv.pdf", f, "application/pdf")})
        assert r.status_code == 201
        app_id = r.json()["application_id"]
        assert hr.get(f"/applications/{app_id}").json()["analysis_status"] == "pending"  # careers never analyses
        assert analyze_pending_once() >= 1
        assert hr.get(f"/applications/{app_id}").json()["analysis_status"] == "done"
