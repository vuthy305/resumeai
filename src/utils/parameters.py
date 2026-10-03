"""Programmatic parameter verification (assignment requires > 10M trainable params)."""
from pathlib import Path

REQUIREMENT = 10_000_000


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def check_parameters(model, model_name: str = "LayoutLMv3-base", report_path=None) -> dict:
    c = count_parameters(model)
    status = "PASS" if c["trainable"] > REQUIREMENT else "FAIL"
    print("========================================")
    print("LAYOUTLMV3 PARAMETER CHECK")
    print("========================================")
    print(f"Model: {model_name}")
    print(f"Total Parameters: {c['total']:,}")
    print(f"Trainable Parameters: {c['trainable']:,}")
    print(f"Non-Trainable Parameters: {c['non_trainable']:,}")
    print("\nRequirement:\nTrainable Parameters > 10,000,000")
    print(f"\nStatus:\n{status}")
    print("========================================")
    if report_path:
        Path(report_path).write_text(
            "ResumeAI LayoutLMv3 Model\n==========================\n\n"
            f"Model:\n{model_name}\n\nTotal Parameters:\n{c['total']:,}\n\n"
            f"Trainable Parameters:\n{c['trainable']:,}\n\n"
            f"Non-Trainable Parameters:\n{c['non_trainable']:,}\n\n"
            "Requirement:\nTrainable Parameters > 10,000,000\n\n"
            f"Status:\n{status}\n", encoding="utf-8")
    if status == "FAIL":
        raise RuntimeError(f"Trainable parameters ({c['trainable']:,}) do not exceed {REQUIREMENT:,}.")
    return {**c, "status": status}
