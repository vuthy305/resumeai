"""Tests run without GPU/PaddleOCR/LayoutLMv3 weights: isolated temp DB + uploads, rule/text-layer fallback."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="resumeai_test_")
os.environ["RESUMEAI_SKIP_MODEL"] = "1"
os.environ["RESUMEAI_DB_URL"] = f"sqlite:///{Path(_TMP, 'test.db').as_posix()}"
os.environ["RESUMEAI_UPLOAD_DIR"] = str(Path(_TMP, "uploads"))
os.environ["RESUMEAI_AUTO_ANALYZE_SECONDS"] = "0"  # tests call analysis explicitly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    from scripts.seed_demo import CANDIDATES, make_cv_pdf
    path = tmp_path_factory.mktemp("cv") / "dara.pdf"
    make_cv_pdf(path, CANDIDATES[0])  # synthetic Backend Developer CV
    return path
