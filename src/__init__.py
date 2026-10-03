"""ResumeAI core package.

Windows fix: PyTorch and PaddlePaddle each ship their own copy of Intel's OpenMP runtime
(libiomp5md.dll). Loading both in one process aborts with "OMP: Error #15". Allowing the duplicate
is the standard workaround for this pair. ResumeAI runs OCR and LayoutLMv3 one after the other
under a lock (never concurrently), which keeps the risk low. Must run before torch/paddle import.
"""
import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")