"""Central configuration for ResumeAI. Every value can be overridden in code or via env vars."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
# Local copy of the HF dataset: files/ = raw repo files, arrow/ = ready-to-load DatasetDict
RAW_DATASET_DIR = RAW_DIR / "layoutlm_resume_data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATASET_DIR = PROCESSED_DIR / "resume_ner"
LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"
DEMO_DIR = DATA_DIR / "demo"
SKILL_DICT_PATH = DATA_DIR / "skill_dictionary.json"
MODELS_DIR = ROOT / "models"
LAYOUTLMV3_DIR = MODELS_DIR / "layoutlmv3"
BEST_MODEL_DIR = LAYOUTLMV3_DIR / "best"
TEXT_ONLY_MODEL_DIR = MODELS_DIR / "layoutlmv3_text_only" / "best"
UPLOAD_DIR = Path(os.getenv("RESUMEAI_UPLOAD_DIR", ROOT / "uploads"))
RESULTS_DIR = ROOT / "results"

DB_URL = os.getenv("RESUMEAI_DB_URL", f"sqlite:///{(ROOT / 'resumeai.db').as_posix()}")
# Two separate websites (separate servers/ports). In production: careers.<bank>.com.kh and an internal HR host.
CAREERS_PORT = int(os.getenv("RESUMEAI_CAREERS_PORT", "8000"))
HR_PORT = int(os.getenv("RESUMEAI_HR_PORT", "8001"))
CAREERS_URL = os.getenv("RESUMEAI_CAREERS_URL", f"http://localhost:{CAREERS_PORT}")
# The HR server analyses new applications in the background every N seconds (0 = off, use the buttons)
AUTO_ANALYZE_SECONDS = float(os.getenv("RESUMEAI_AUTO_ANALYZE_SECONDS", "10"))

DATASET_NAME = "Kunling/layoutlm_resume_data"
BASE_MODEL = "microsoft/layoutlmv3-base"

ENTITY_TYPES = ["NAME", "EMAIL", "PHONE", "ADDRESS", "SECTIONHEADER"]
# Fallback label order, only used if the dataset does not ship label names.
DEFAULT_LABELS = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ("B", "I")]

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_MB = 10
PDF_DPI = 170          # 150-200 keeps memory low while OCR stays readable
PDF_MAX_PAGES = 5      # CVs rarely exceed this; protects memory on huge PDFs

# Skip loading LayoutLMv3/PaddleOCR (tests, CPU-only demos). The pipeline then
# falls back to PDF text layer + rule-based extraction and says so in the output.
SKIP_MODEL = os.getenv("RESUMEAI_SKIP_MODEL", "0") == "1"
OCR_USE_GPU = os.getenv("RESUMEAI_OCR_GPU", "0") == "1"  # CPU OCR keeps 4 GB VRAM for LayoutLMv3


@dataclass
class TrainConfig:
    base_model: str = BASE_MODEL
    output_dir: str = str(LAYOUTLMV3_DIR)
    epochs: int = 5
    batch_size: int = 1
    gradient_accumulation: int = 8
    max_length: int = 256
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    gradient_checkpointing: bool = True
    image_size: int = 224          # must match the pretrained LayoutLMv3 patch grid
    num_workers: int = 0           # 0 is safest on Windows
    val_ratio: float = 0.1
    seed: int = 42
    no_layout: bool = False        # ablation: zero boxes + blank image
    log_every: int = 20
    max_train_docs: int = 0        # 0 = all (use small values for smoke tests)


@dataclass
class PreprocessConfig:
    max_side: int = 2000
    grayscale: bool = False
    contrast: bool = False
    contrast_factor: float = 1.3
    denoise: bool = False
    sharpen: bool = False
    deskew: bool = False


DEFAULT_MATCH_WEIGHTS = {
    "required_skills": 0.50,
    "experience": 0.25,
    "education": 0.15,
    "preferred_skills": 0.10,
}

COMPANY_NAME = os.getenv("RESUMEAI_COMPANY_NAME", "Tonle Bank")
JOB_STATUSES = ["Open", "Closed"]
EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Contract", "Internship"]
APPLICATION_STATUSES = ["Submitted", "Under Review", "Shortlisted", "Interview", "Rejected"]

for _d in (RAW_DIR, PROCESSED_DIR, DEMO_DIR, LAYOUTLMV3_DIR, UPLOAD_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
