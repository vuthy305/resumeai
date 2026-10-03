"""Seed SQLite with 3 jobs, 12 SYNTHETIC demo candidates (generated PDF CVs) and applications.
All demo people are fictional. Usage:
    python scripts/seed_demo.py --reset            # jobs + candidates + applications
    python scripts/seed_demo.py --reset --analyze  # also run the CV pipeline + matching now
"""
import argparse
from datetime import datetime, timedelta
import shutil
import zlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from src.config import DEMO_DIR, UPLOAD_DIR
from src.database.db import SessionLocal, engine, init_db
from src.database.models import Application, Base, Candidate, Job

JOBS = [
    dict(title="Backend Developer", department="Information Technology",
         description="Design, build and maintain secure REST APIs and databases for the bank's digital channels.",
         required_skills=["Python", "PostgreSQL", "REST API"], preferred_skills=["Docker", "FastAPI"],
         minimum_experience=2, education_requirement="Bachelor's degree in Computer Science or related field",
         location="Phnom Penh", employment_type="Full-time", closing_days=30),
    dict(title="Data Analyst", department="Business Intelligence",
         description="Analyse banking data, build dashboards and support data-driven decisions.",
         required_skills=["SQL", "Python", "Excel"], preferred_skills=["Power BI", "Tableau"],
         minimum_experience=1, education_requirement="Bachelor's degree in Statistics, Economics, Computer Science or related field",
         location="Phnom Penh", employment_type="Full-time", closing_days=21),
    dict(title="Credit Officer", department="Credit & Lending",
         description="Assess loan applications, analyse borrower financials and manage credit risk.",
         required_skills=["Credit Analysis", "Financial Analysis", "Excel"], preferred_skills=["Banking", "Risk Assessment"],
         minimum_experience=2, education_requirement="Bachelor's degree in Finance, Accounting, Economics or related field",
         location="Siem Reap", employment_type="Full-time", closing_days=45),
]

# name, city, summary, experience[(title, company, dates, detail)], education[(degree, school, dates)],
# skills line, languages, certifications, applies_to
CANDIDATES = [
    ("Dara Vann", "Phnom Penh, Cambodia", "Backend developer with a focus on reliable banking APIs.",
     [("Backend Developer", "Mekong Digital Co., Ltd", "Mar 2021 - Present", "Built RESTful APIs with Python and FastAPI on PostgreSQL; containerised services with Docker."),
      ("Junior Developer", "Angkor Soft Solutions", "Jan 2019 - Feb 2021", "Maintained Django web applications and wrote SQL reports.")],
     [("Bachelor of Science in Computer Science", "Royal University of Phnom Penh", "2014 - 2018")],
     "Python, FastAPI, Django, PostgreSQL, Docker, Git, Linux, REST API", "Khmer, English", ["AWS Certified Cloud Practitioner"], ["Backend Developer"]),
    ("Maly Sok", "Phnom Penh, Cambodia", "Software engineer building web services.",
     [("Software Engineer", "Lotus Technology Co., Ltd", "Jun 2022 - Present", "Developed Node.js and Python microservices with MySQL.")],
     [("Bachelor of Engineering in Information Technology", "Institute of Technology of Cambodia", "2017 - 2021")],
     "Python, Node.js, MySQL, REST API, Git, Docker", "Khmer, English, French", [], ["Backend Developer"]),
    ("Rithy Chea", "Siem Reap, Cambodia", "Java developer moving into Python backend work.",
     [("Java Developer", "Delta Systems", "Jan 2018 - Dec 2022", "Built Spring Boot services and Oracle Database procedures."),
      ("Backend Engineer", "Kampuchea Fintech Group", "Jan 2023 - Present", "Designed REST APIs in Python and PostgreSQL for payments.")],
     [("Master of Science in Software Engineering", "Build Bright University", "2016 - 2018")],
     "Java, Spring Boot, Python, PostgreSQL, Oracle Database, REST API, Kubernetes", "Khmer, English", [], ["Backend Developer", "Data Analyst"]),
    ("Sophea Lim", "Phnom Penh, Cambodia", "Frontend developer with some API experience.",
     [("Frontend Developer", "Pixel Studio Co., Ltd", "Aug 2023 - Present", "Built React interfaces consuming REST APIs.")],
     [("Associate Degree in Computer Science", "National Polytechnic Institute", "2020 - 2022")],
     "JavaScript, React, HTML, CSS, REST API, Git", "Khmer, English", [], ["Backend Developer"]),
    ("Nita Keo", "Phnom Penh, Cambodia", "Data analyst turning banking data into decisions.",
     [("Data Analyst", "Golden River Bank", "Feb 2022 - Present", "Wrote SQL queries, automated reports with Python and pandas, built Power BI dashboards."),
      ("Reporting Assistant", "Sunrise Microfinance", "Jan 2021 - Jan 2022", "Prepared monthly Excel reports.")],
     [("Bachelor of Science in Statistics", "Royal University of Phnom Penh", "2016 - 2020")],
     "SQL, Python, Pandas, Excel, Power BI, Statistics", "Khmer, English", ["Microsoft Certified: Power BI Data Analyst Associate"], ["Data Analyst"]),
    ("Vuthy Heng", "Battambang, Cambodia", "Business analyst comfortable with spreadsheets and visualisation.",
     [("Business Analyst", "Emerald Trading Co., Ltd", "May 2023 - Present", "Built Tableau dashboards and Excel models.")],
     [("Bachelor of Business Administration", "University of Battambang", "2018 - 2022")],
     "Excel, Tableau, SQL, Data Visualization", "Khmer, English, Thai", [], ["Data Analyst"]),
    ("Chenda Pov", "Phnom Penh, Cambodia", "Recent graduate interested in analytics.",
     [("Data Intern", "Blue Ocean Research", "Jun 2025 - Dec 2025", "Cleaned survey data in Excel and Python.")],
     [("Bachelor of Economics", "National University of Management", "2021 - 2025")],
     "Excel, Python, Statistics", "Khmer, English", [], ["Data Analyst", "Credit Officer"]),
    ("Sokunthea Ly", "Phnom Penh, Cambodia", "Credit professional with strong financial statement analysis.",
     [("Credit Officer", "Golden River Bank", "Jan 2020 - Present", "Performed credit analysis and financial analysis for SME loans; conducted risk assessment."),
      ("Loan Officer", "Sunrise Microfinance", "Mar 2017 - Dec 2019", "Handled loan origination and collection.")],
     [("Bachelor of Business Administration in Finance", "National University of Management", "2012 - 2016")],
     "Credit Analysis, Financial Analysis, Risk Assessment, Excel, Banking, KYC", "Khmer, English", [], ["Credit Officer"]),
    ("Bopha Nhem", "Kampong Cham, Cambodia", "Accountant seeking a move into lending.",
     [("Accountant", "Riverside Holdings", "Jul 2021 - Present", "Prepared financial statements and performed financial analysis in Excel.")],
     [("Bachelor of Accounting", "Build Bright University", "2016 - 2020")],
     "Accounting, Financial Analysis, Excel", "Khmer, English", ["ACCA Part-qualified"], ["Credit Officer"]),
    ("Kosal Tep", "Phnom Penh, Cambodia", "Banking operations officer.",
     [("Teller", "Capital Commercial Bank", "Jan 2024 - Present", "Retail banking operations and customer service.")],
     [("Diploma in Banking and Finance", "Cambodia Banking Institute", "2021 - 2023")],
     "Banking, Customer Service, Excel", "Khmer", [], ["Credit Officer"]),
    ("Pisey Mao", "Phnom Penh, Cambodia", "Full-stack developer with data skills.",
     [("Full Stack Developer", "Orchid Solutions", "Sep 2020 - Present", "Built Flask REST APIs on PostgreSQL, wrote SQL analytics and Excel exports.")],
     [("Bachelor of Science in Information Technology", "Paragon International University", "2016 - 2020")],
     "Python, Flask, PostgreSQL, SQL, REST API, Excel, Docker", "Khmer, English", [], ["Backend Developer", "Data Analyst"]),
    ("Samnang Ouk", "Takeo, Cambodia", "IT support technician.",
     [("IT Support Technician", "Metro Services", "Jan 2022 - Present", "Helpdesk support, networking and Windows troubleshooting.")],
     [("High School Diploma", "Takeo High School", "2018 - 2021")],
     "IT Support, Networking, Troubleshooting, Linux", "Khmer", [], ["Backend Developer"]),
]


def _wrap(text, width, fontsize, font="helv"):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if fitz.get_text_length(trial, fontname=font, fontsize=fontsize) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def make_cv_pdf(path: Path, c):
    name, city, summary, exps, edus, skills, langs, certs, _ = c
    email = name.lower().replace(" ", ".") + "@example.com"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y, left, width = 60, 50, 495

    def line(text, size=10, font="helv", gap=4):
        nonlocal y
        for l in _wrap(text, width, size, font):
            page.insert_text((left, y), l, fontsize=size, fontname=font)
            y += size + gap

    def header(text):
        nonlocal y
        y += 8
        line(text.upper(), 12, "hebo", 6)

    line(name, 20, "hebo", 8)
    line(city)
    line(f"{email}  |  +855 12 {zlib.crc32(name.encode()) % 900 + 100} {zlib.crc32(city.encode() + name.encode()) % 900 + 100}")
    header("Summary"); line(summary)
    header("Work Experience")
    for title, company, dates, detail in exps:
        line(title, 11, "hebo"); line(f"{company}  |  {dates}"); line(detail); y += 4
    header("Education")
    for degree, school, dates in edus:
        line(degree, 11, "hebo"); line(f"{school}, {dates}")
    header("Skills"); line(skills)
    header("Languages"); line(langs)
    if certs:
        header("Certifications")
        for ce in certs:
            line(ce)
    doc.save(str(path))
    doc.close()
    return email


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop all tables first")
    ap.add_argument("--analyze", action="store_true", help="run CV pipeline + matching now")
    args = ap.parse_args()
    if args.reset:
        Base.metadata.drop_all(engine)
    init_db()
    db = SessionLocal()
    if db.query(Job).count():
        print("Database already seeded. Use --reset to re-seed.")
        return
    jobs = {}
    for j in JOBS:
        j = dict(j)
        days = j.pop("closing_days", None)
        if days:
            j["closing_date"] = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        job = Job(**j)
        db.add(job)
        jobs[j["title"]] = job
    db.flush()
    n_apps = 0
    for c in CANDIDATES:
        fname = c[0].replace(" ", "_") + "_CV.pdf"
        demo_path = DEMO_DIR / fname
        email = make_cv_pdf(demo_path, c)
        stored = f"demo_{fname}"
        shutil.copy(demo_path, UPLOAD_DIR / stored)
        cand = Candidate(name=c[0], email=email, phone=None, cv_filename=fname, cv_path=stored)
        db.add(cand)
        db.flush()
        for title in c[-1]:
            db.add(Application(job_id=jobs[title].id, candidate_id=cand.id))
            n_apps += 1
    db.commit()
    print(f"Seeded {len(jobs)} jobs, {len(CANDIDATES)} synthetic candidates, {n_apps} applications.")
    print(f"Synthetic CV PDFs: {DEMO_DIR}")
    if args.analyze:
        from api.services import analyze_application, get_manager
        m = get_manager()
        print(f"Analyzing with core method: {m.pipeline.core_method}")
        for app in db.query(Application).all():
            analyze_application(db, app)
            print(f"  app {app.id:>2} {app.candidate.name:<16} -> {app.job.title:<18} "
                  f"score={app.match_score} [{app.analysis_status}]")
    db.close()


if __name__ == "__main__":
    main()
