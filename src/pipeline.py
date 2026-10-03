"""CV file -> structured profile.

PDF/Image -> preprocessing -> PaddleOCR -> LayoutLMv3 (core entities) -> rule-based supplementary fields.
Fallbacks (always reported in the output, never silent):
  * no trained LayoutLMv3 checkpoint  -> rule-based baseline for core entities
  * PaddleOCR not installed           -> PDF text layer (real word boxes from PyMuPDF); images fail clearly
  * DOCX                              -> python-docx text + rules (no layout)
Pages are processed sequentially under a lock to protect 4 GB of VRAM.
"""
import threading
import time
from pathlib import Path

from src.config import IMAGE_EXTENSIONS
from src.document_processing.docx_processor import extract_docx_text
from src.document_processing.image_processor import ImagePreprocessor, load_image
from src.document_processing.pdf_processor import extract_text_layer_words, iter_pdf_pages
from src.extraction.baseline import RuleBasedExtractor
from src.extraction.entities import merge_page_fields
from src.extraction.profile_builder import ProfileBuilder
from src.utils.logging_utils import get_logger
from src.utils.text_utils import find_emails, find_phones, group_lines

log = get_logger("pipeline")
CORE_FIELDS = ["name", "email", "phone", "address", "section_headers"]
SUPPLEMENTARY_FIELDS = ["skills", "education", "degree", "experience", "total_experience_years",
                        "job_titles", "companies", "certifications", "languages"]


class PipelineError(ValueError):
    """User-facing processing error (message is safe to show)."""


class CVPipeline:
    def __init__(self, ocr=None, extractor=None, preprocessor: ImagePreprocessor | None = None):
        self.ocr = ocr
        self.extractor = extractor
        self.pre = preprocessor or ImagePreprocessor()
        self.baseline = RuleBasedExtractor()
        self.builder = ProfileBuilder()
        self.lock = threading.Lock()

    @property
    def core_method(self) -> str:
        return "layoutlmv3" if self.extractor else "rule_based_baseline"

    def _core(self, image, words, boxes, page):
        if self.extractor:
            return self.extractor.extract_from_ocr(image, {"words": words, "boxes": boxes}, page)
        return self.baseline.extract(words, boxes, page)

    def _process_page(self, image, page, text_layer=None):
        image = self.pre(image)
        if self.ocr is not None:
            ocr = self.ocr.extract(image)
            words, boxes, lines, engine = ocr["words"], ocr["boxes"], ocr["lines"], "paddleocr"
        elif text_layer is not None:
            words, boxes, engine = text_layer["words"], text_layer["boxes"], "pdf_text_layer"
            lines = [" ".join(words[i] for i in grp) for grp in group_lines(boxes)]
        else:
            raise PipelineError("OCR engine is not available on the server, so image CVs cannot be read.")
        fields = self._core(image, words, boxes, page) if words else merge_page_fields([])
        fields.pop("labels", None)
        return fields, lines, engine, len(words)

    def process_file(self, path) -> dict:
        path = Path(path)
        ext = path.suffix.lower()
        t0 = time.time()
        with self.lock:
            if ext == ".docx":
                text = extract_docx_text(path)
                core = self.baseline.extract_from_text(text)
                core_method, engine, n_pages = "rule_based_text (docx)", "python-docx", 1
            elif ext == ".pdf" or ext in IMAGE_EXTENSIONS:
                page_fields, all_lines, engines, n_words = [], [], set(), 0
                if ext == ".pdf":
                    layers = extract_text_layer_words(path) if self.ocr is None else None
                    for page, image in iter_pdf_pages(path):
                        tl = layers[page - 1] if layers else None
                        f, lines, eng, n = self._process_page(image, page, tl)
                        del image
                        page_fields.append(f); all_lines += lines; engines.add(eng); n_words += n
                else:
                    f, lines, eng, n = self._process_page(load_image(path), 1)
                    page_fields.append(f); all_lines += lines; engines.add(eng); n_words += n
                if n_words == 0:
                    raise PipelineError("No text could be read from this CV (empty or unreadable document).")
                core = merge_page_fields(page_fields)
                text = "\n".join(all_lines)
                core_method, engine, n_pages = self.core_method, "+".join(sorted(engines)), len(page_fields)
            else:
                raise PipelineError(f"Unsupported file format: {ext}")

        fallback = []
        if not core.get("email") and find_emails(text):
            core["email"] = find_emails(text)[0]; fallback.append("email")
        if not core.get("phone") and find_phones(text):
            core["phone"] = find_phones(text)[0]; fallback.append("phone")
        profile = self.builder.build(text, core)
        return {
            "profile": profile,
            "core_entities": {k: core.get(k) for k in CORE_FIELDS},
            "evidence": core.get("evidence", []),
            "extracted_text": text,
            "extraction_method": core_method,
            "ocr_engine": engine,
            "pages_processed": n_pages,
            "field_sources": {
                "core_entities": {"method": core_method, "fields": CORE_FIELDS},
                "rule_based_supplementary": SUPPLEMENTARY_FIELDS,
                "regex_fallback": fallback,
            },
            "processing_seconds": round(time.time() - t0, 2),
        }
