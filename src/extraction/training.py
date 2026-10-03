"""Low-VRAM LayoutLMv3 fine-tuning and evaluation (shared by scripts and notebooks)."""
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import LABEL_MAP_PATH, PROCESSED_DATASET_DIR, RESULTS_DIR, TrainConfig
from src.extraction.layoutlmv3_model import (ChunkDataset, autocast_ctx, build_model, load_processor,
                                             predict_words)
from src.extraction.metrics import seqeval_rows
from src.utils.hardware import gpu_memory_str, print_hardware
from src.utils.parameters import check_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed():
    from datasets import load_from_disk
    if not Path(PROCESSED_DATASET_DIR).exists():
        raise FileNotFoundError("Processed dataset not found. Run: python scripts/prepare_dataset.py")
    ds = load_from_disk(str(PROCESSED_DATASET_DIR))
    labels = json.loads(Path(LABEL_MAP_PATH).read_text())["labels"]
    return ds, labels


def split_train_val(ds, val_ratio: float, seed: int):
    s = ds["train"].train_test_split(test_size=val_ratio, seed=seed)
    return s["train"], s["test"]


def evaluate_docs(model, processor, docs, device, max_length=256, use_amp=True, no_layout=False):
    model.eval()
    y_true, y_pred = [], []
    for doc in docs:
        y_pred.append(predict_words(model, processor, doc["image"], doc["words"], doc["boxes"],
                                    device, max_length, use_amp, no_layout))
        y_true.append(list(doc["labels"]))
    return y_true, y_pred


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    hw = print_hardware()
    device = hw["device"]
    use_amp = cfg.fp16 and device == "cuda"
    ds, label_list = load_processed()
    label2id = {l: i for i, l in enumerate(label_list)}
    train_ds, val_ds = split_train_val(ds, cfg.val_ratio, cfg.seed)
    if cfg.max_train_docs:
        train_ds = train_ds.select(range(min(cfg.max_train_docs, len(train_ds))))
        val_ds = val_ds.select(range(min(max(cfg.max_train_docs // 4, 2), len(val_ds))))
    print(f"Train docs: {len(train_ds)} | Val docs: {len(val_ds)} | Test docs (untouched): {len(ds['test'])}")
    print(f"Config: {asdict(cfg)}")

    processor = load_processor(cfg.base_model)
    model = build_model(label_list, cfg.base_model)
    report = RESULTS_DIR / ("parameter_report.txt" if not cfg.no_layout else "parameter_report_text_only.txt")
    params = check_parameters(model, "LayoutLMv3-base (token classification)", report)
    if cfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing: enabled")
        except Exception as e:
            print(f"Gradient checkpointing unavailable: {e}")
    model.to(device)

    train_data = ChunkDataset(train_ds, processor, label2id, cfg.max_length, cfg.no_layout)
    loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    print(f"Training chunks (<= {cfg.max_length} tokens): {len(train_data)}")

    from transformers import get_linear_schedule_with_warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_updates = math.ceil(len(loader) / cfg.gradient_accumulation) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_updates), total_updates)
    scaler = _make_scaler(use_amp)

    out_dir = Path(cfg.output_dir)
    best_dir = out_dir / "best"
    best_f1, history = -1.0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx(device, use_amp):
                loss = model(**batch).loss / cfg.gradient_accumulation
            scaler.scale(loss).backward()
            running += loss.item() * cfg.gradient_accumulation
            if step % cfg.gradient_accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f} | {gpu_memory_str()}")
        y_true, y_pred = evaluate_docs(model, processor, val_ds, device, cfg.max_length, use_amp, cfg.no_layout)
        rows = seqeval_rows(y_true, y_pred)
        val_f1 = next(r["f1"] for r in rows if r["entity"] == "MACRO")
        rec = {"epoch": epoch, "train_loss": round(running / max(len(loader), 1), 4), "val_macro_f1": val_f1,
               "seconds": round(time.time() - t0, 1), "gpu": gpu_memory_str()}
        history.append(rec)
        print(f"== epoch {epoch}: {rec}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            (best_dir / "train_config.json").write_text(json.dumps({**asdict(cfg), "best_val_macro_f1": best_f1,
                                                                    "epoch": epoch, "parameters": params}, indent=2))
            print(f"   saved best checkpoint -> {best_dir}")
    name = "training_log_text_only.json" if cfg.no_layout else "training_log.json"
    (RESULTS_DIR / name).write_text(json.dumps(history, indent=2))
    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return best_dir


def evaluate_checkpoint(model_dir, split: str = "test", max_length: int = 256, fp16: bool = True,
                        no_layout: bool = False) -> tuple[list[dict], list, list]:
    from src.extraction.layoutlmv3_model import load_trained
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, _ = load_processed()
    model, processor = load_trained(model_dir, device)
    t0 = time.time()
    y_true, y_pred = evaluate_docs(model, processor, ds[split], device, max_length, fp16 and device == "cuda", no_layout)
    print(f"Evaluated {len(y_true)} docs in {time.time() - t0:.1f}s on {device}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return seqeval_rows(y_true, y_pred), y_true, y_pred
