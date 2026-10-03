"""LayoutLMv3Extractor: image + OCR words/boxes -> BIO -> NAME/EMAIL/PHONE/ADDRESS/SECTIONHEADER."""
import torch

from src.config import BEST_MODEL_DIR
from src.extraction.entities import bio_to_entities, entities_to_fields
from src.extraction.layoutlmv3_model import load_trained, predict_words
from src.utils.logging_utils import get_logger

log = get_logger("inference")


class LayoutLMv3Extractor:
    def __init__(self, model_dir=BEST_MODEL_DIR, device: str | None = None, fp16: bool = True,
                 max_length: int = 256, ocr=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = fp16 and self.device == "cuda"
        self.max_length = max_length
        self.ocr = ocr
        log.info(f"Loading LayoutLMv3 from {model_dir} on {self.device} (fp16={self.use_amp})")
        self.model, self.processor = load_trained(model_dir, self.device)

    def predict_labels(self, image, words, boxes) -> list[str]:
        return predict_words(self.model, self.processor, image, words, boxes, self.device,
                             self.max_length, self.use_amp)

    def extract_from_ocr(self, image, ocr_result: dict, page: int = 1) -> dict:
        labels = self.predict_labels(image, ocr_result["words"], ocr_result["boxes"])
        fields = entities_to_fields(bio_to_entities(ocr_result["words"], labels, page))
        fields["labels"] = labels
        return fields

    def extract(self, image, page: int = 1) -> dict:
        if self.ocr is None:
            from src.document_processing.ocr import OCRProcessor
            self.ocr = OCRProcessor()
        return self.extract_from_ocr(image, self.ocr.extract(image), page)
