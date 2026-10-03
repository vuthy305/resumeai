import numpy as np
from PIL import Image

from src.document_processing.ocr import OCRProcessor, parse_paddle_result, split_line_into_words
from src.utils.text_utils import clip_box_1000, normalize_box


def test_normalize_box_range_and_order():
    assert normalize_box([100, 50, 300, 150], 1000, 500) == [100, 100, 300, 300]
    assert normalize_box([300, 150, 100, 50], 1000, 500) == [100, 100, 300, 300]  # swapped corners
    assert normalize_box([-5, 0, 2000, 900], 1000, 500) == [0, 0, 1000, 1000]     # clipped
    assert clip_box_1000([1200, -3, 5, 7]) == [5, 0, 1000, 7]


def test_split_line_boxes_stay_inside_line_box():
    parts = split_line_into_words("John Smith", [0, 10, 100, 30])
    assert [w for w, _ in parts] == ["John", "Smith"]
    for _, (x0, y0, x1, y1) in parts:
        assert 0 <= x0 < x1 <= 100 and (y0, y1) == (10, 30)


def test_parse_paddle_v2_and_v3_formats():
    v2 = [[[[[0, 0], [50, 0], [50, 10], [0, 10]], ("Hello world", 0.99)]]]
    v3 = [{"rec_texts": ["Hello world"], "rec_scores": [0.99],
           "rec_polys": [np.array([[0, 0], [50, 0], [50, 10], [0, 10]])]}]
    for raw in (v2, v3):
        lines = parse_paddle_result(raw)
        assert lines[0]["text"] == "Hello world" and lines[0]["box"] == [0, 0, 50, 10]
    assert parse_paddle_result([None]) == []


class FakeEngine:  # mimics PaddleOCR 2.x .ocr()
    def ocr(self, arr, cls=False):
        return [[[[[20, 10], [220, 10], [220, 40], [20, 40]], ("John Smith", 0.98)],
                 [[[20, 60], [300, 60], [300, 80], [20, 80]], ("john@example.com", 0.95)],
                 [[[20, 90], [100, 90], [100, 100], [20, 100]], ("noise", 0.2)]]]


def test_ocr_processor_output_format():
    out = OCRProcessor(engine=FakeEngine()).extract(Image.new("RGB", (400, 200), "white"))
    assert set(["words", "boxes", "scores"]) <= set(out)
    assert out["words"] == ["John", "Smith", "john@example.com"]   # low-score line filtered
    assert len(out["words"]) == len(out["boxes"]) == len(out["scores"])
    assert all(0 <= v <= 1000 for b in out["boxes"] for v in b)
