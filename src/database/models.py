from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    department: Mapped[str] = mapped_column(String(200), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    required_skills: Mapped[list] = mapped_column(JSON, default=list)
    preferred_skills: Mapped[list] = mapped_column(JSON, default=list)
    minimum_experience: Mapped[float] = mapped_column(Float, default=0)
    education_requirement: Mapped[str] = mapped_column(String(300), default="")
    status: Mapped[str] = mapped_column(String(30), default="Open")
    location: Mapped[str] = mapped_column(String(200), default="Phnom Penh")
    employment_type: Mapped[str] = mapped_column(String(50), default="Full-time")
    closing_date: Mapped[str | None] = mapped_column(String(20), nullable=True)  # YYYY-MM-DD
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    applications: Mapped[list["Application"]] = relationship(back_populates="job", cascade="all, delete-orphan")

    def to_matching_dict(self) -> dict:
        return {"title": self.title, "department": self.department, "description": self.description,
                "required_skills": self.required_skills or [], "preferred_skills": self.preferred_skills or [],
                "minimum_experience_years": self.minimum_experience or 0,
                "education_requirement": self.education_requirement or ""}


class Candidate(Base):
    __tablename__ = "candidates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(200), index=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    cv_filename: Mapped[str] = mapped_column(String(300))
    cv_path: Mapped[str] = mapped_column(String(500))       # stored file name inside UPLOAD_DIR only
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_profile_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    applications: Mapped[list["Application"]] = relationship(back_populates="candidate")


class Application(Base):
    __tablename__ = "applications"
    __table_args__ = (UniqueConstraint("job_id", "candidate_id", name="uq_job_candidate"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), index=True)
    match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_skills: Mapped[list] = mapped_column(JSON, default=list)
    missing_skills: Mapped[list] = mapped_column(JSON, default=list)
    experience_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    education_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    match_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    analysis_status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|done|failed
    analysis_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="Submitted")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    job: Mapped[Job] = relationship(back_populates="applications")
    candidate: Mapped[Candidate] = relationship(back_populates="applications")
