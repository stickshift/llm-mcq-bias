import torch

__all__ = [
    "torch_device",
]


def torch_device() -> torch.device:
    """Configure gpus."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
