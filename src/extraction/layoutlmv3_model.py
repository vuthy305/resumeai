"""LayoutLMv3 helpers: loading, word-level chunking, encoding and word-level prediction.

CVs are longer than max_length (256 by default for 4 GB VRAM), so every document is split
into consecutive word chunks that fit in max_length tokens. Each chunk is encoded with the
same page image. Word-level predictions use the first sub-token of each word.
"""
import contextlib

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from src.config import BASE_MODEL
from src.utils.text_utils import clip_box_1000

BLANK_BOX = [0, 0, 0, 0]


def load_processor(path_or_name=BASE_MODEL) -> LayoutLMv3Processor:
    return LayoutLMv3Processor.from_pretrained(str(path_or_name), apply_ocr=False)


def build_model(label_list: list[str], base_model: str = BASE_MODEL) -> LayoutLMv3ForTokenClassification:
    id2label = {i: l for i, l in enumerate(label_list)}
    return LayoutLMv3ForTokenClassification.from_pretrained(
        base_model, num_labels=len(label_list), id2label=id2label, label2id={l: i for i, l in id2label.items()})


def load_trained(model_dir, device: str):
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_dir)).to(device).eval()
    return model, load_processor(model_dir)


def word_token_counts(tokenizer, words: list[str]) -> list[int]:
    if not words:
        return []
    enc = tokenizer(text=words, boxes=[BLANK_BOX] * len(words), add_special_tokens=False, truncation=False)
    counts = [0] * len(words)
    for wi in enc.word_ids():
        if wi is not None:
            counts[wi] += 1
    return counts


def chunk_ranges(counts: list[int], max_length: int) -> list[tuple[int, int]]:
    budget = max_length - 2  # <s> and </s>
    ranges, start, used = [], 0, 0
    for i, c in enumerate(counts):
        c = max(c, 1)
        if used + c > budget and i > start:
            ranges.append((start, i))
            start, used = i, 0
        used += c
    if start < len(counts):
        ranges.append((start, len(counts)))
    return ranges


def prepare_inputs(image: Image.Image, boxes: list[list[int]], no_layout: bool):
    """Ablation switch: no_layout removes 2D layout (zero boxes) and visual content (blank image)."""
    if no_layout:
        return Image.new("RGB", (224, 224), "white"), [BLANK_BOX] * len(boxes)
    return image.convert("RGB"), [clip_box_1000(b) for b in boxes]


def encode_chunk(tokenizer, pixel_values, words, boxes, label_ids=None, max_length: int = 256):
    enc = tokenizer(text=words, boxes=boxes, word_labels=label_ids, truncation=True,
                    padding="max_length", max_length=max_length, return_tensors="pt")
    enc["pixel_values"] = pixel_values
    return enc


def autocast_ctx(device: str, enabled: bool):
    if device == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict_words(model, processor, image, words, boxes, device: str, max_length: int = 256,
                  use_amp: bool = True, no_layout: bool = False) -> list[str]:
    if not words:
        return []
    id2label = model.config.id2label
    image, boxes = prepare_inputs(image, boxes, no_layout)
    pixel_values = processor.image_processor(image, return_tensors="pt")["pixel_values"]
    preds = ["O"] * len(words)
    for s, e in chunk_ranges(word_token_counts(processor.tokenizer, words), max_length):
        enc = encode_chunk(processor.tokenizer, pixel_values, words[s:e], boxes[s:e], None, max_length)
        word_ids = enc.word_ids(0)
        inputs = {k: v.to(device) for k, v in enc.items()}
        with autocast_ctx(device, use_amp):
            logits = model(**inputs).logits[0]
        ids = logits.argmax(-1).tolist()
        seen = set()
        for ti, wi in enumerate(word_ids):
            if wi is None or wi in seen:
                continue
            seen.add(wi)
            preds[s + wi] = id2label[ids[ti]]
        del inputs, logits
    return preds


class ChunkDataset(torch.utils.data.Dataset):
    """Training dataset over (document, word-chunk) pairs. Encodes lazily to save RAM."""

    def __init__(self, hf_ds, processor, label2id: dict, max_length: int = 256, no_layout: bool = False):
        self.ds, self.processor, self.label2id = hf_ds, processor, label2id
        self.max_length, self.no_layout = max_length, no_layout
        self.items = []
        for d_idx, words in enumerate(hf_ds["words"]):
            for s, e in chunk_ranges(word_token_counts(processor.tokenizer, words), max_length):
                self.items.append((d_idx, s, e))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d_idx, s, e = self.items[i]
        doc = self.ds[d_idx]
        image, boxes = prepare_inputs(doc["image"], doc["boxes"], self.no_layout)
        pixel_values = self.processor.image_processor(image, return_tensors="pt")["pixel_values"]
        labels = [self.label2id[l] for l in doc["labels"][s:e]]
        enc = encode_chunk(self.processor.tokenizer, pixel_values, doc["words"][s:e], boxes[s:e], labels, self.max_length)
        return {k: v.squeeze(0) for k, v in enc.items()}
