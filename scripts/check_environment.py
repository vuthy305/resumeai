"""Print versions + hardware and test critical imports.  python scripts/check_environment.py"""
import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.hardware import print_hardware  # noqa: E402


def version(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", getattr(m, "VersionBind", "installed"))
    except Exception as e:
        return f"NOT AVAILABLE ({type(e).__name__}: {str(e)[:80]})"


print(f"Python version:        {platform.python_version()} ({platform.system()})")
for label, mod in [("PyTorch", "torch"), ("Transformers", "transformers"), ("Datasets", "datasets"),
                   ("PaddlePaddle", "paddle"), ("PaddleOCR", "paddleocr"), ("PyMuPDF", "pymupdf"),
                   ("FastAPI", "fastapi"), ("SQLAlchemy", "sqlalchemy"),
                   ("seqeval", "seqeval")]:
    print(f"{label + ' version:':<23}{version(mod)}")
print_hardware()

checks = {}
try:
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor  # noqa: F401
    checks["LayoutLMv3 import"] = "OK"
except Exception as e:
    checks["LayoutLMv3 import"] = f"FAIL: {e}"
try:
    from paddleocr import PaddleOCR  # noqa: F401
    checks["PaddleOCR import"] = "OK"
except Exception as e:
    checks["PaddleOCR import"] = f"FAIL: {e}"
for name, mod in [("FastAPI import", "fastapi")]:
    try:
        importlib.import_module(mod)
        checks[name] = "OK"
    except Exception as e:
        checks[name] = f"FAIL: {e}"
try:
    from src.database.db import check_connection, init_db
    init_db()
    checks["SQLite connection"] = "OK" if check_connection() else "FAIL"
except Exception as e:
    checks["SQLite connection"] = f"FAIL: {e}"
print("\nIMPORT / CONNECTION TESTS")
for k, v in checks.items():
    print(f"  {k:<20} {v}")
sys.exit(0 if all(v == "OK" for v in checks.values()) else 1)
