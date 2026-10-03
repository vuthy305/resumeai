from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.config import APPLICATION_STATUSES


class JobCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    department: str = ""
    description: str = ""
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    minimum_experience_years: float = Field(0, ge=0, le=50)
    education_requirement: str = ""
    location: str = "Phnom Penh"
    employment_type: str = "Full-time"
    closing_date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    status: str = "Open"


class JobUpdate(BaseModel):
    """Every field optional: send only what changed."""
    title: str | None = Field(None, min_length=2, max_length=200)
    department: str | None = None
    description: str | None = None
    required_skills: list[str] | None = None
    preferred_skills: list[str] | None = None
    minimum_experience_years: float | None = Field(None, ge=0, le=50)
    education_requirement: str | None = None
    location: str | None = None
    employment_type: str | None = None
    closing_date: str | None = Field(None, pattern=r"^(\d{4}-\d{2}-\d{2})?$")
    status: str | None = None


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    department: str
    description: str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float
    education_requirement: str
    status: str
    created_at: datetime
    applicant_count: int = 0


class StatusUpdate(BaseModel):
    status: str

    def validate_status(self):
        if self.status not in APPLICATION_STATUSES:
            raise ValueError(f"Status must be one of {APPLICATION_STATUSES}")


class HealthOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    device: str
    ocr_loaded: bool
    extraction_method: str
    notes: list[str] = []
