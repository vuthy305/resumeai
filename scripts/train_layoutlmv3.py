"""Fine-tune LayoutLMv3-base on the resume NER dataset (low-VRAM defaults).
    python scripts/train_layoutlmv3.py --epochs 5 --batch-size 1 --gradient-accumulation 8 --max-length 256 --learning-rate 5e-5 --fp16
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LAYOUTLMV3_DIR, TrainConfig  # noqa: E402


def parse_args(argv=None) -> TrainConfig:
    d = TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--gradient-accumulation", type=int, default=d.gradient_accumulation)
    ap.add_argument("--max-length", type=int, default=d.max_length)
    ap.add_argument("--learning-rate", type=float, default=d.learning_rate)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=d.fp16)
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=d.gradient_checkpointing)
    ap.add_argument("--num-workers", type=int, default=d.num_workers)
    ap.add_argument("--val-ratio", type=float, default=d.val_ratio)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--output-dir", default=str(LAYOUTLMV3_DIR))
    ap.add_argument("--no-layout", action="store_true", help="ablation: zero boxes + blank image")
    ap.add_argument("--max-train-docs", type=int, default=0, help="smoke test on N docs")
    a = ap.parse_args(argv)
    return TrainConfig(epochs=a.epochs, batch_size=a.batch_size, gradient_accumulation=a.gradient_accumulation,
                       max_length=a.max_length, learning_rate=a.learning_rate, fp16=a.fp16,
                       gradient_checkpointing=a.gradient_checkpointing, num_workers=a.num_workers,
                       val_ratio=a.val_ratio, seed=a.seed, output_dir=a.output_dir, no_layout=a.no_layout,
                       max_train_docs=a.max_train_docs)


if __name__ == "__main__":
    from src.extraction.training import train
    best = train(parse_args())
    print(f"\nBest checkpoint: {best}")
