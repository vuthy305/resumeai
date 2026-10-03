"""Hardware detection with CPU fallback."""
import platform


def get_hardware_info() -> dict:
    info = {"cpu": platform.processor() or platform.machine(), "gpu": None, "vram_gb": None,
            "cuda": False, "cuda_version": None, "pytorch": None, "device": "cpu"}
    try:
        import torch
        info["pytorch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(gpu=props.name, vram_gb=round(props.total_memory / 1024 ** 3, 2),
                        cuda=True, cuda_version=torch.version.cuda, device="cuda")
    except ImportError:
        pass
    return info


def print_hardware(info: dict | None = None) -> dict:
    info = info or get_hardware_info()
    print("=================================")
    print("HARDWARE")
    print("=================================")
    print(f"CPU:     {info['cpu']}")
    print(f"GPU:     {info['gpu'] or 'Not available (CPU fallback)'}")
    print(f"VRAM:    {str(info['vram_gb']) + ' GB' if info['vram_gb'] else 'N/A'}")
    print(f"CUDA:    {info['cuda']} ({info['cuda_version'] or '-'})")
    print(f"PyTorch: {info['pytorch'] or 'not installed'}")
    print("=================================")
    return info


def gpu_memory_str() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 1024 ** 2
            p = torch.cuda.max_memory_allocated() / 1024 ** 2
            return f"GPU mem {a:.0f} MB (peak {p:.0f} MB)"
    except ImportError:
        pass
    return "GPU mem n/a (CPU)"
